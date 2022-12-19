import argparse
import sys
import os

import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import einops

from torchvision import datasets, transforms, utils

from tqdm import tqdm

from data_loader import SYSUData
from loss import TripletLoss
from model import ModelAdaptive, ModelAdaptive_Deep
from reid_tools import validate

from vqvae_deep import VQVAE_Deep as VQVAE
from scheduler import CycleScheduler
import distributed as dist
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

invTrans = transforms.Compose([
    transforms.Normalize(mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
    transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
])

def random_pair(args):
    l = np.arange(args.batch_size) * args.num_pos
    l = l[:, None]
    r = np.random.randint(1, args.num_pos, args.batch_size).reshape(-1, 1)
    ids = (np.tile(np.arange(args.num_pos), args.batch_size).reshape(-1, args.num_pos) + r) % args.num_pos + l
    ids = ids.reshape(-1)
    return ids

def train(epoch, loader, model, optimizer, scheduler, device, optimizer_reid):
    if dist.is_primary():
        loader = tqdm(loader)

    criterion = nn.MSELoss()
    triplet_criterion = TripletLoss()

    latent_loss_weight = 0.25
    sample_size = 25

    mse_sum = 0
    mse_n = 0

    id_sum = 0
    feat_sum = 0

    ir_sum = 0

    for i, (img1, img2, label1, label2, camera1, camera2) in enumerate(loader):
        # vq_vae, person_id =  model['vq_vae'], model['person_id']

        img1 = img1.to(device)
        img2 = img2.to(device)
        label1 = label1.to(device)
        label2 = label2.to(device)

        labels = torch.cat((label1, label2), 0)

        bs = img1.size(0)

        model.person_id.requires_grad_(True)
        model.person_id.train()
        feat, score, feat2d, actMap, feat2d_x3 = model.encode_person(img1)
        m = actMap.view(bs, -1).median(dim=1)[0].view(bs, 1, 1, 1)
        zeros = actMap < m

        actMap = torch.ones_like(actMap).cuda()
        actMap[zeros] = 0

        upMask = F.upsample(actMap, scale_factor=16, mode='bilinear')


        loss_id_real = torch.nn.functional.cross_entropy(score, label1)
        loss_triplet, _ = triplet_criterion(feat, label1)
        var = einops.rearrange(feat, '(n p) ... -> n p ...', p=args.num_pos).var(dim=1)
        mean = einops.rearrange(feat, '(n p) ... -> n p ...', p=args.num_pos).mean(dim=1)

        optimizer_reid.zero_grad()
        loss_Re = loss_id_real + loss_triplet + var.mean()
        loss_Re.backward()
        optimizer_reid.step()

        ids = random_pair(args)

        # assign_adain_params(adain_params[:bs], model.adaptor)

        w = torch.rand(bs, 3).cuda() + 0.01
        w = w / w.sum(dim=1, keepdim=True)
        gray = torch.einsum('b c w h, b c -> b w h', img1, w).unsqueeze(1).expand(-1, 3, -1, -1)

        # gray = img2

        rgb_b, rgb_t = model.encode_content(img1)
        rgb_content_itself, latent_loss = model.quantize_content(rgb_b, rgb_t)
        rgb_reconst = model.decode(rgb_content_itself)

        gray_b, gray_t = model.encode_content(gray)
        gray_b_f, gray_t_f = model.fuse(gray_b, gray_t, feat2d_x3 * actMap ,feat2d * actMap)
        gray_content_itself, latent_loss_gray = model.quantize_content(gray_b_f, gray_t_f)
        rgb_fake = model.decode(gray_content_itself)

        gray_b_other, gray_t_other = model.fuse(gray_b, gray_t, feat2d_x3[ids] * actMap[ids], feat2d[ids] * actMap[ids])
        gray_content_other, latent_loss_other = model.quantize_content(gray_b_other, gray_t_other)
        rgb_fake_other = model.decode(gray_content_other)

        ir_b, ir_t = model.encode_content(img2)
        ir_b_f, ir_t_f = model.fuse(ir_b, ir_t, feat2d_x3 * actMap, feat2d * actMap)
        ir_content, latent_loss_ir = model.quantize_content(ir_b_f, ir_t_f)
        ir_fake = model.decode(ir_content)

        model.person_id.requires_grad_(False)
        model.person_id.eval()
        featIR, score, _, _, _ = model.person_id(xRGB=ir_fake, xIR=None, modal=1, with_feature=True)
        loss_id_real_ir = torch.nn.functional.cross_entropy(score, label2)
        loss_feat_ir = criterion(featIR, feat.detach())
        loss_Re_Ir = loss_id_real_ir + loss_feat_ir

        maskImg = img1*upMask

        recon_loss = criterion(rgb_reconst*upMask, maskImg) +\
                     criterion(rgb_fake*upMask, maskImg) +\
                     criterion(rgb_fake_other*upMask, maskImg)
        recon_loss_feat = criterion(gray_content_itself, rgb_content_itself) +\
                          criterion(gray_content_other, rgb_content_itself)
        latent_loss = (latent_loss + latent_loss_gray + latent_loss_other + latent_loss_ir).mean()
        loss_G = (recon_loss_feat +  recon_loss + latent_loss_weight * latent_loss)  # + loss_id_fake + feat_loss + loss_kl_fake



        optimizer.zero_grad()
        (loss_G + loss_Re_Ir).backward()
        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        part_mse_sum = recon_loss.item() * img1.shape[0]
        part_mse_n = img1.shape[0]
        comm = {"mse_sum": part_mse_sum, "mse_n": part_mse_n}
        comm = dist.all_gather(comm)

        for part in comm:
            mse_sum += part["mse_sum"]
            mse_n += part["mse_n"]

        id_err = loss_Re #loss_id_fake.item() + loss_id_real.item() + loss_triplet.item() + loss_kl_fake
        id_sum += id_err
        feat_err = recon_loss_feat.item()
        feat_sum += feat_err
        ir_sum += loss_Re_Ir.item()

        if dist.is_primary():
            lr = optimizer.param_groups[0]["lr"]

            loader.set_description(
                (
                    f"e: {epoch + 1}; mse: {recon_loss.item():.5f}({mse_sum / mse_n:.5f}); "
                    f"lat: {latent_loss.item():.3f}; "
                    f"id: {id_err:.3f}({id_sum / (i+1):.3f}); "
                    f"ft: {feat_err:.3f}({feat_sum / (i+1):.5f}); "
                    f"ir: {loss_Re_Ir.item():.3f}({ir_sum / (i + 1):.5f}); "
                )
            )

            if i % 100 == 0:
                # model.eval()
                index = np.random.choice(np.arange(bs), min(bs, sample_size), replace=False)

                sample = img1[index]
                fake_recon = rgb_reconst[index]
                fake_rgb = rgb_fake[index]
                real_ir = img2[index]
                gray = gray[index]

                fake_rgb_other = rgb_fake_other[index]

                # with torch.no_grad():
                #     out, _ = model(sample)
                # model.train()

                utils.save_image(
                    invTrans(torch.cat([sample, fake_rgb, fake_rgb_other,
                                        real_ir, ir_fake[index],
                                        2 * (upMask[index].expand(-1, 3, -1, -1)) - 1], 0)),
                    f"sample-deep-transfer/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png",
                    nrow=len(sample),
                    # normalize=True,
                    range=(-1, 1),
                )

    writer.add_scalar("ID_loss/train", id_sum / len(loader), epoch)
    writer.add_scalar("Rec_loss/train", mse_sum / len(loader), epoch)
    writer.add_scalar("Feat_loss/train", feat_sum / len(loader), epoch)
    writer.add_scalar("ID_ir/train", ir_sum / len(loader), epoch)

def main(args):
    device = "cuda"

    args.distributed = dist.get_world_size() > 1

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            # transforms.CenterCrop(args.size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = SYSUData(args.path, transform=transform)
    loader_batch = args.batch_size * args.num_pos
    vq_vae = VQVAE().to(device)
    model = ModelAdaptive_Deep(dataset.num_class, vq_vae).to(device)

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
        )

    if len(args.resume) > 0:
        model_path = args.resume
        if os.path.isfile(model_path):
            print('==> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint, strict=False)
            print('==> loaded checkpoint {} (epoch)'
                  .format(args.resume))
        else:
            print('==> no checkpoint found at {}'.format(args.resume))

    ignored_params = list(map(id, model.person_id.parameters()))
    ids = set(map(id, model.parameters()))
    params = filter(lambda p: id(p) in ids, model.parameters())
    base_params = filter(lambda p: id(p) not in ignored_params, params)

    optimizer_reID = optim.Adam(model.person_id.parameters(), lr=args.lr)
    optimizer = optim.Adam(base_params , lr=args.lr)

    scheduler = None
    if args.sched == "cycle":
        scheduler = CycleScheduler(
            optimizer,
            args.lr,
            n_iter=len(dataset) * args.epoch,
            momentum=None,
            warmup_proportion=0.05,
        )


    for i in range(args.start, args.epoch):
        sampler = dataset.samplize(args.batch_size, args.num_pos)
        loader = DataLoader(
            dataset, batch_size=loader_batch // args.n_gpu, sampler=sampler, num_workers=args.workers
        )


        train(i, loader, model, optimizer, scheduler, device, optimizer_reID)
        if i % 4 == 0:
            validate(0, model.person_id, args=args)
        model.person_id.train()
        torch.save(model.state_dict(), f"checkpoint-deep-transfer/vqvae_last.pt")
        if i % 10 == 0 and dist.is_primary():
            torch.save(model.state_dict(), f"checkpoint-deep-transfer/vqvae_{str(i + 1).zfill(3)}.pt")





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=1)

    port = (
        2 ** 15
        + 2 ** 14
        + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    )
    parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")

    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--epoch", type=int, default=560)
    parser.add_argument("--start", "-s", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--sched", type=str)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_pos", type=int, default=4)
    parser.add_argument('--resume', '-r', default='', type=str,
                        help='resume from checkpoint')
    parser.add_argument('--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument("--path", type=str, default='../Datasets/SYSU-MM01/')

    args = parser.parse_args()

    print(args)

    dist.launch(main, args.n_gpu, 1, 0, args.dist_url, args=(args,))
