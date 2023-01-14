import argparse
import random
import sys
import os

import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import einops

from torchvision import datasets, transforms, utils
import torchvision.transforms as T

from tqdm import tqdm

from data_loader import SYSUData
from loss import TripletLoss, TripletLoss_WRT
from model import ModelAdaptive, ModelAdaptive_Deep, embed_net
from old_model import embed_net2
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

stage_reconstruction = 40

class RandomCropBoxes:
    def __init__(self, n, size):
        self.n = n
        self.size = size

    def __call__(self, imgs):
        h, w = self.size, self.size
        H, W = imgs.shape[-2:]
        for img in imgs:
            y, x = np.random.randint(0, H - h, self.n), np.random.randint(0, W - w, self.n)
            for xx,yy in zip(x,y):
                img[:, yy:yy + h, xx:xx + w] = random.random()
        return imgs


aug_transforms = transforms.Compose([
    # T.ColorJitter(brightness=.15, hue=.13),
    T.ElasticTransform(alpha=25.0),
    RandomCropBoxes(n=15, size=20)
])

aug_transforms_rec = transforms.Compose([
    # T.ColorJitter(brightness=.15, hue=.13),
    T.ElasticTransform(alpha=25.0),
    RandomCropBoxes(n=10, size=10)
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
    triplet_criterion = TripletLoss_WRT()

    latent_loss_weight = 0.25
    sample_size = 16

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

        aug_rgb = aug_transforms(img1)
        aug_ir = aug_transforms(img2)
        bs = img1.size(0)

        ir_b, ir_t = model.encode_content(img2)
        ir_content_itself, latent_loss = model.quantize_content(ir_b, ir_t)
        ir_reconst = model.decode(ir_content_itself).expand(-1, 3, -1, -1)
        recon_loss = criterion(ir_reconst, img2)
        loss_G = (recon_loss + latent_loss_weight * latent_loss)
        loss_feat_ir = loss_Re_Ir = loss_Re = torch.Tensor([-1])

        if epoch > stage_reconstruction :
            model.person_id.requires_grad_(True)
            model.person_id.train()
            feat, score, feat2d, actMap, feat2d_x3 = model.person_id(xRGB=aug_rgb, xIR=aug_ir, modal=0, with_feature=True)
            featV, featI = torch.split(feat, bs)
            # m = actMap.view(feat.shape[0], -1).median(dim=1)[0].view(feat.shape[0], 1, 1, 1)
            # zeros = actMap < (m - 0.1)
            # ones = actMap > (m + 0.02)
            # actMap[zeros] = 0
            # actMap[ones] = 1
            upMask = F.upsample(actMap, scale_factor=16, mode='bilinear')

            loss_id_real = torch.nn.functional.cross_entropy(score, labels)
            loss_triplet = triplet_criterion(featV, label1)[0] + triplet_criterion(featI, label2)[0]
            Feat = einops.rearrange(feat, '(m n p) ... -> n (p m) ...', p=args.num_pos, m=feat.shape[0] // img1.shape[0])
            # var = Feat.var(dim=1)
            # mean = Feat.mean(dim=1)



            w = torch.rand(bs, 3).cuda() + 0.01
            w = w / (abs(w.sum(dim=1, keepdim=True)) + 0.01)
            gray = torch.einsum('b c w h, b c -> b w h', img1, w).unsqueeze(1).expand(-1, 3, -1, -1)
            invIndex = np.random.choice(bs, bs // 2, replace=False)
            gray[invIndex] = 1 - gray[invIndex]

            rgb_b, rgb_t = model.encode_content(gray)
            rgb_b_f, rgb_t_f = rgb_b, rgb_t#model.fuse(rgb_b, rgb_t, feat2d_x3[bs:] , feat2d[bs:])
            rgb_content, latent_loss_ir = model.quantize_content(rgb_b_f, rgb_t_f)
            inter = model.decode(rgb_content).expand(-1,3,-1,-1)

            feat_fake, score_fake, _, _, _ = model.person_id(xRGB = None, xZ=inter.detach(), xIR=ir_reconst.detach(), modal=0, with_feature=True)
            loss_id_fake = torch.nn.functional.cross_entropy(score_fake, labels)
            loss_triplet_fake, _ = triplet_criterion(feat_fake, labels)
            modal_free_loss = criterion(feat_fake, feat)
            loss_fake = loss_id_fake + modal_free_loss

            optimizer_reid.zero_grad()
            loss_Re = loss_id_real + loss_triplet + loss_fake #+ var.mean()
            loss_Re.backward()
            optimizer_reid.step()

            model.person_id.requires_grad_(False)
            model.person_id.eval()
            featG, score, _, _, _ = model.person_id(xRGB=None, xIR=inter, modal=2, with_feature=True)
            loss_id_real_ir = torch.nn.functional.cross_entropy(score, label1)

            FV = einops.rearrange(feat[:bs].detach(), '(m n p) ... -> n (p m) ...', p=args.num_pos, m=1) # reshaped to b * p
            sV = FV.sum(dim=1, keepdim=True) # sum of features for each person
            centerV = (sV - FV) / (args.num_pos - 1) # make centers of others for each person

            FG = einops.rearrange(featG, '(m n p) ... -> n (p m) ...', p=args.num_pos, m=1)  # reshaped to b * p
            sG = FG.sum(dim=1, keepdim=True)  # sum of features for each person
            centerG_X = (sG - FG) / (args.num_pos - 1)  # make centers of others for each person

            # centerV = einops.rearrange(feat[:bs], '(m n p) ... -> n (p m) ...', p=args.num_pos, m=1).mean(dim=1)
            centerT = einops.rearrange(feat[bs:], '(m n p) ... -> n (p m) ...', p=args.num_pos, m=1).mean(dim=1)
            centerG = FG.mean(dim=1)

            pos = (centerG - centerT.detach()).pow(2).sum(dim=1)
            neg = (centerG_X - centerV.detach()).pow(2).sum(dim=-1).mean(dim=1)
            loss_feat_ir = F.margin_ranking_loss(pos, neg, -1 * torch.ones_like(pos), margin=0.1) #criterion(featG, feat[bs:].detach())
            loss_Re_Ir = loss_id_real_ir + loss_feat_ir

            # recon_loss_feat = criterion(gray_content_itself, rgb_content_itself) +\
            #                   criterion(gray_content_other, rgb_content_itself)
            latent_loss = latent_loss + latent_loss_ir
            loss_G = loss_G + 0.1 * loss_Re_Ir
              # + loss_id_fake + feat_loss + loss_kl_fake


        optimizer.zero_grad()
        loss_G.backward()
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

        id_err = loss_Re.item() #loss_id_fake.item() + loss_id_real.item() + loss_triplet.item() + loss_kl_fake
        id_sum += id_err
        feat_err = loss_feat_ir.item()
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

                rgb = aug_rgb[index]
                ir = aug_ir[index]
                ir_rec = ir_reconst[index]
                rgb2ir = inter[index] if epoch > stage_reconstruction else img2[index]
                g = gray[index] if epoch > stage_reconstruction else rgb

                # with torch.no_grad():
                #     out, _ = model(sample)
                # model.train()

                utils.save_image(
                    invTrans(torch.cat([rgb, g, rgb2ir, ir, ir_rec], 0)),
                    f"sample-deep-transfer/ir50_{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png",
                    nrow=len(rgb),
                    # normalize=True,
                    range=(-1, 1),
                )

    writer.add_scalar("ID_loss/train", id_sum / len(loader), epoch)
    writer.add_scalar("Rec_loss/train", mse_sum / len(loader), epoch)
    writer.add_scalar("Feat_loss/train", feat_sum / len(loader), epoch)
    writer.add_scalar("ID_ir/train", ir_sum / len(loader), epoch)


def main(args):
    device = "cuda"
    best_mAP = 0
    best_i = 0
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
    vq_vae = VQVAE(out_channel=1).to(device)
    model = ModelAdaptive_Deep(dataset.num_class, vq_vae, arch='resnet50').to(device)
    model.person_id = embed_net2(dataset.num_class).to(device)
    # checkpoint = torch.load(
    #     '/home/mahdi/PycharmProjects/vq-vae-2-pytorch/sysu_att_p8_n4_lr_0.03_seed_0_gray_randChanU2_best.t')
    # model.person_id.load_state_dict(checkpoint['net'])
    model.person_id.to(device)

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

            if "net" not in checkpoint:
                model.load_state_dict(checkpoint, strict=False)
                # model.person_id = embed_net(dataset.num_class, 'off', 'off', arch='resnet50').to(device)
            else:
                best_mAP = checkpoint['mAP']
                best_i = checkpoint['epoch']
                model.load_state_dict(checkpoint["net"], strict=True)
            print(f'==> loaded checkpoint {args.resume} (epoch {best_i} mAP {best_mAP})')
        else:
            print('==> no checkpoint found at {}'.format(args.resume))

    ignored_params = list(map(id, model.person_id.parameters()))
    ids = set(map(id, model.parameters()))
    params = filter(lambda p: id(p) in ids, model.parameters())
    base_params = filter(lambda p: id(p) not in ignored_params, params)


    optimizer = optim.Adam(base_params , lr=args.lr)

    ignored_params_reid = list(map(id, model.person_id.bottleneck.parameters())) \
                          + list(map(id, model.person_id.classifier.parameters()))

    base_params_reid = filter(lambda p: id(p) not in ignored_params_reid, model.person_id.parameters())

    optimizer_reID = optim.SGD([
        {'params': base_params_reid, 'lr': 0.1 * 0.001},
        {'params': model.person_id.bottleneck.parameters(), 'lr': 0.001},
        {'params': model.person_id.classifier.parameters(), 'lr': 0.001}
    ], weight_decay=5e-4, momentum=0.9, nesterov=True)

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
        if i >= stage_reconstruction and i % 4 == 0:
            mAP = validate(0, model, args=args, mode='all')
            if mAP > best_mAP:
                best_mAP = mAP
                best_i = i
                obj = {
                    "mAP" : best_mAP,
                    "epoch": best_i,
                    "net": model.state_dict()
                }
                torch.save(obj, f"checkpoint-deep-transfer/vqvae_ir50Z_best.pt")
            print('best mAP: {:.2%}| epoch: {}'.format(best_mAP, best_i))

        model.person_id.train()
        torch.save(model.state_dict(), f"checkpoint-deep-transfer/vqvae_ir50Z_last.pt")
        if i % 10 == 0 and dist.is_primary():
            torch.save(model.state_dict(), f"checkpoint-deep-transfer/vqvae_ir50Z_{str(i + 1).zfill(3)}.pt")





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
