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
    ranking_loss = nn.MarginRankingLoss(margin=5)

    mse_sum = 0
    mse_n = 0

    loss = 0
    id_sum = 0
    feat_sum = 0

    ir_sum = 0
    correct = 0

    eigen_inter, eigen_intra, svd_sum = 0 , 0, 0


    model.person_id.requires_grad_(True)
    model.person_id.train()

    for i, (img1, img2, label1, label2, camera1, camera2) in enumerate(loader):
        # vq_vae, person_id =  model['vq_vae'], model['person_id']

        img1 = img1.to(device)
        img2 = img2.to(device)
        label1 = label1.to(device)
        label2 = label2.to(device)
        labels = torch.cat((label1, label2), 0)
        bs = img1.size(0)

        feat, score, feat2d, actMap, feat2d_x3 = model.person_id(xRGB=img1, xIR=img2, modal=0, with_feature=True)

        _, predicted = score.max(1)
        correct += (predicted.eq(labels).sum().item())

        loss_id_real = torch.nn.functional.cross_entropy(score, labels)
        loss_triplet, _ = triplet_criterion(feat, labels)

        F = einops.rearrange(feat, '(m n p) ... -> n (p m) ...', p=args.num_pos, m=2)
        var = F.var(dim=1)
        mean = F.mean(dim=1)

        _, S_inter, _ = torch.pca_lowrank(mean, q=args.batch_size, center=True, niter=3)
        _, S_intra, _ = torch.pca_lowrank(F, q=2 * args.num_pos, center=True, niter=3)
        # eigen_inter += S_inter.mean().item()
        # eigen_intra += S_intra.mean().item()
        svd_loss = ranking_loss(S_inter.mean(), S_intra.mean(), torch.tensor(1))
        svd_sum += svd_loss.item()

        optimizer_reid.zero_grad()
        loss_Re = loss_id_real + loss_triplet + S_intra.mean() + svd_loss #+ var.mean()
        loss_Re.backward()
        optimizer_reid.step()


        part_mse_sum = 0
        mse_n += 2 * img1.shape[0]

        loss += loss_Re.item()
        id_err = loss_id_real.item() #loss_id_fake.item() + loss_id_real.item() + loss_triplet.item() + loss_kl_fake
        id_sum += id_err
        feat_err = loss_triplet.item()
        feat_sum += feat_err
        ir_sum += 0

        if dist.is_primary():
            lr = optimizer.param_groups[0]["lr"]

            loader.set_description(
                (
                    f"e: {epoch + 1}; l: {loss_Re.item():.3f}({loss / (i+1):.5f}); "
                    f"id: {loss_id_real.item():.3f};({id_sum / (i+1):.5f}); "
                    f"tr: {feat_err:.3f}({feat_sum / (i+1):.5f}); "
                    f"svd: {svd_loss.item():.3f}({svd_sum / (i+1):.5f}); "
                    f"p: ({correct * 100 / mse_n:.2f}); "

                )
            )


    writer.add_scalar("ID_loss/train", id_sum / len(loader), epoch)
    writer.add_scalar("Rec_loss/train", mse_sum / len(loader), epoch)
    writer.add_scalar("Feat_loss/train", feat_sum / len(loader), epoch)
    writer.add_scalar("svd/train", svd_sum / len(loader), epoch)


def main(args):
    device = "cuda"

    args.distributed = dist.get_world_size() > 1

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            # transforms.CenterCrop(args.size),
            transforms.Pad(10),
            transforms.RandomCrop((args.img_h, args.img_w)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = SYSUData(args.path, transform=transform)
    loader_batch = args.batch_size * args.num_pos
    vq_vae = VQVAE().to(device)
    model = ModelAdaptive_Deep(dataset.num_class, vq_vae, arch='resnet50').to(device)

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
            try:
                model.person_id.load_state_dict(checkpoint, strict=True)
                print('==> loaded checkpoint {} (epoch)'
                      .format(args.resume))
            except:
                print('==> loaded checkpoint failure from {}'.format(args.resume))
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

    best_mAp , best_epoch = 0 , 0
    for i in range(args.start, args.epoch):
        sampler = dataset.samplize(args.batch_size, args.num_pos)
        loader = DataLoader(
            dataset, batch_size=loader_batch // args.n_gpu, sampler=sampler, num_workers=args.workers
        )


        train(i, loader, model, optimizer, scheduler, device, optimizer_reID)
        if i % 4 == 0:
            mAp = validate(0, model.person_id, args=args, mode='all')
            writer.add_scalar("mAP/eval", mAp, i)
            if mAp > best_mAp:
                torch.save(model.person_id.state_dict(), f"checkpoint/reid_best.pt")
                best_epoch = i
                best_mAp = mAp
            print("best mAP {:.2%} epoch {}".format(best_mAp, best_epoch))

        model.person_id.train()
        torch.save(model.person_id.state_dict(), f"checkpoint/reid_last.pt")
        if i % 10 == 0 and dist.is_primary():
            torch.save(model.person_id.state_dict(), f"checkpoint/reid_{str(i + 1).zfill(3)}.pt")





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
    parser.add_argument("--batch_size", "-b", type=int, default=4)
    parser.add_argument("--num_pos", "-p", type=int, default=8)
    parser.add_argument('--resume', '-r', default='', type=str,
                        help='resume from checkpoint')
    parser.add_argument('--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument("--path", type=str, default='../Datasets/SYSU-MM01/')
    parser.add_argument('--gpu', default='0', type=str,
                        help='gpu device ids for CUDA_VISIBLE_DEVICES')

    parser.add_argument('--img_w', default=144, type=int,
                        metavar='imgw', help='img width')
    parser.add_argument('--img_h', default=288, type=int,
                        metavar='imgh', help='img height')

    args = parser.parse_args()

    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    dist.launch(main, args.n_gpu, 1, 0, args.dist_url, args=(args,))
