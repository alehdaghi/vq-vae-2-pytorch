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

from vqvae_deep import VQVAE
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


        rgb_fake, latent_loss = model(img1)

        # assign_adain_params(adain_params[:bs], model.adaptor)

        recon_loss = criterion(rgb_fake, img1)
        latent_loss = latent_loss.mean()
        loss_G = (recon_loss + latent_loss_weight * latent_loss) #+ loss_id_fake + feat_loss + loss_kl_fake
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

        id_err = 0 #loss_id_fake.item() + loss_id_real.item() + loss_triplet.item() + loss_kl_fake
        id_sum += id_err
        feat_err = 0#feat_loss.item()
        feat_sum += feat_err

        if dist.is_primary():
            lr = optimizer.param_groups[0]["lr"]

            loader.set_description(
                (
                    f"epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; "
                    f"lat: {latent_loss.item():.3f}({mse_sum / mse_n:.5f}); "
                    f"id: {id_err:.3f}({id_sum / (i+1):.3f}); "
                    f"feat: {feat_err:.3f}({feat_sum / (i+1):.5f}); "
                    f"lr: {lr:.5f}"
                )
            )

            if i % 100 == 0:
                # model.eval()

                index = np.random.choice(np.arange(bs), min(bs, sample_size), replace=False)

                sample = img1[index]
                fake_rgb = rgb_fake[index]
                # fake_ir = ir_fake[index]
                real_ir = img2[index]

                # with torch.no_grad():
                #     out, _ = model(sample)

                utils.save_image(
                    invTrans(torch.cat([sample, fake_rgb], 0)),
                    f"sample-deep/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png",
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



    model = VQVAE().to(device)

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

    optimizer_reID = None #optim.Adam(model.person_id.parameters(), lr=args.lr)
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
        torch.save(model.state_dict(), f"checkpoint-deep/vqvae_last.pt")
        if i % 10 == 0 and dist.is_primary():
            torch.save(model.state_dict(), f"checkpoint-deep/vqvae_{str(i + 1).zfill(3)}.pt")


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
