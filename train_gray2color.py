import argparse
import sys
import os

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import einops

from torchvision import datasets, transforms, utils

from tqdm import tqdm

from data_loader import SYSUData
from loss import TripletLoss
from model import ModelAdaptive, assign_adain_params
from vqvae import VQVAE
from scheduler import CycleScheduler
import distributed as dist

invTrans = transforms.Compose([
    transforms.Normalize(mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
    transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
])

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


    for i, (img1, img2, label1, label2, camera1, camera2) in enumerate(loader):
        model.zero_grad()

        img1 = img1.to(device)
        img2 = img2.to(device)
        label1 = label1.to(device)
        label2 = label2.to(device)

        labels = torch.cat((label1, label2), 0)

        bs = img1.size(0)

        # model.person_id.requires_grad_(True)

        # feat, score, feat2d, actMap = model.encode_person(img1)
        feat2d = model.encode_style(img1)

        # ids = np.arange(bs).reshape(-1, args.num_pos)
        # list(map(np.random.shuffle, ids))

        l = np.arange(args.batch_size) * args.num_pos
        l = l[:, None]
        r = np.random.randint(1, args.num_pos, args.batch_size).reshape(-1, 1)
        ids = (np.tile(np.arange(args.num_pos), args.batch_size).reshape(-1, args.num_pos) + r) % args.num_pos + l
        ids = ids.reshape(-1)




        img1_other = img1[ids]
        feat2d_other = model.encode_style(img1_other)

        # loss_id_real = torch.nn.functional.cross_entropy(score, label1)
        # loss_triplet, _ = triplet_criterion(feat, label1)
        # var  = einops.rearrange(feat, '(n p) ... -> n p ...', p=args.num_pos).var(dim=1)
        # mean = einops.rearrange(feat, '(n p) ... -> n p ...', p=args.num_pos).mean(dim=1)
        # loss_Re = loss_id_real + loss_triplet + var.mean()

        # model.person_id.requires_grad_(False)

        # adain_params = model.mlp(feat.detach())
        # assign_adain_params(adain_params[bs:], model.adaptor)
        w = torch.rand(bs, 3).cuda() + 0.01
        w = w / w.sum(dim=1, keepdim=True)
        gray = torch.einsum('b c w h, b c -> b w h', img1, w).unsqueeze(1).expand(-1, 3, -1, -1)
        gray = img2


        rgb_content, latent_loss = model.encode_content(img1)
        rgb_reconst = model.decode(rgb_content)

        gray_content, _ = model.encode_content(gray)

        gray_content_itself = model.fuse(gray_content, feat2d)
        rgb_fake = model.decode(gray_content_itself)

        gray_content_other = model.fuse(gray_content, feat2d_other)
        rgb_fake_other = model.decode(gray_content_other)


        # feat_ir_fake, score_ir_fake = model.person_id(xRGB=None, xIR=ir_fake, modal=2)

        # mean_fake = einops.rearrange(feat_ir_fake, '(n p) ... -> n p ...', p=args.num_pos).mean(dim=1)

        # loss_id_fake = torch.nn.functional.cross_entropy(score_ir_fake, label2)
        # loss_kl_fake = 100 * torch.nn.functional.kl_div(score_ir_fake.log_softmax(dim=1), score.detach().softmax(dim=1))
        # feat_loss = criterion(mean.detach(), mean_fake)

        # assign_adain_params(adain_params[:bs], model.adaptor)

        recon_loss = criterion(rgb_reconst, img1) + criterion(rgb_fake, img1) + criterion(rgb_fake_other, img1)
        recon_loss_feat = criterion(gray_content_itself, rgb_content) + criterion(gray_content_other, rgb_content)
        latent_loss = latent_loss.mean()
        loss_G = (recon_loss_feat + recon_loss + latent_loss_weight * latent_loss) #+ loss_id_fake + feat_loss + loss_kl_fake


        optimizer.zero_grad()
        # optimizer_reid.zero_grad()
        # (loss_Re + loss_G).backward()
        loss_G.backward()
        # optimizer_reid.step()


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

        id_err = 0#loss_id_fake.item() + loss_id_real.item() + loss_triplet.item() + loss_kl_fake
        id_sum += id_err
        feat_err = recon_loss_feat.item()
        feat_sum += feat_err

        if dist.is_primary():
            lr = optimizer.param_groups[0]["lr"]

            loader.set_description(
                (
                    f"epoch: {epoch + 1};"
                    f" mse: {recon_loss.item():.5f}({mse_sum / mse_n:.5f}); "
                    f"lat: {latent_loss.item():.3f}; "
                    f"id: {id_err:.3f}({id_sum / (i+1):.3f}); "
                    f"feat: {feat_err:.3f}({feat_sum / (i+1):.5f}); "
                    f"lr: {lr:.5f}"
                )
            )

            if i % 100 == 0:
                # model.eval()

                index = np.random.choice(np.arange(bs), min(bs, sample_size), replace=False)

                sample = img1[index]
                fake_recon = rgb_reconst[index]
                fake_rgb = rgb_fake[index]
                real_ir = gray[index]

                fake_rgb_other = rgb_fake_other[index]


                # with torch.no_grad():
                #     out, _ = model(sample)

                utils.save_image(
                    invTrans(torch.cat([sample, fake_rgb, real_ir, img1_other[index] ,fake_rgb_other], 0)),
                    f"sample/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png",
                    nrow=len(sample),
                    # normalize=True,
                    range=(-1, 1),
                )

                model.train()


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



    model = ModelAdaptive(dataset.num_class).to(device) #VQVAE().to(device)

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
            model.load_state_dict(checkpoint)
            print('==> loaded checkpoint {} (epoch)'
                  .format(args.resume))
        else:
            print('==> no checkpoint found at {}'.format(args.resume))

    optimizer_reID = optim.Adam(model.person_id.parameters(), lr=args.lr)
    optimizer = optim.Adam(list(model.parameters()) , lr=args.lr)

    scheduler = None
    if args.sched == "cycle":
        scheduler = CycleScheduler(
            optimizer,
            args.lr,
            n_iter=len(dataset) * args.epoch,
            momentum=None,
            warmup_proportion=0.05,
        )

    for i in range(args.epoch):
        sampler = dataset.samplize(args.batch_size, args.num_pos)
        loader = DataLoader(
            dataset, batch_size=loader_batch // args.n_gpu, sampler=sampler, num_workers=args.workers
        )

        train(i, loader, model, optimizer, scheduler, device, optimizer_reID)

        torch.save(model.state_dict(), f"checkpoint/vqvae_last.pt")
        if i % 10 == 0 and dist.is_primary():
            torch.save(model.state_dict(), f"checkpoint/vqvae_{str(i + 1).zfill(3)}.pt")


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
