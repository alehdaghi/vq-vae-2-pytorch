import argparse
import os
from pycocotools.coco import COCO
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# import torchvision.models.detection.mask_rcnn as mask_rcnn
from vision.engine import *
import distributed as dist
import cv2
import numpy as np
import resource
from PIL import Image
# from ssd.utils import *

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
trans = transforms.Compose([
    transforms.ToTensor(),
    #     normalize,
])


class annToTarget():
    def __init__(self, coco=None):
        self.coco = coco

    def __call__(self, ann):
        N = len(ann)
        boxes = torch.FloatTensor(N, 4)
        labels = torch.LongTensor(N)
        image_id = torch.LongTensor(1)
        area = torch.IntTensor(N)
        iscrowd = torch.IntTensor(N)
        masks = []  # torch.empty((N, 600, 400), dtype=torch.uint8)
        i = 0
        for obj in ann:
            if obj['bbox'][2] < 1 or obj['bbox'][3] < 1:
                N = N - 1
                continue

            boxes[i] = torch.FloatTensor(obj['bbox'])
            labels[i] = obj['category_id']
            image_id[0] = obj['image_id']
            area[i] = obj['area']
            iscrowd[i] = obj['iscrowd']
            i = i+1
            # masks.append(torch.from_numpy(self.coco.annToMask(obj)))

        boxes[:, 2:] += boxes[:, :2] # xywh -> xyXY

        dict = {
            'boxes': boxes[:N],
            'labels': labels[:N] - 1,
            'image_id': image_id,
            "area": area[:N],
            "iscrowd": iscrowd[:N],
            # "masks": torch.stack(masks) if N > 0 else torch.empty((N, 300, 200))

        }
        return dict


def collate_fn(batch):
    return tuple(zip(*batch))

img_h, img_w = 400 , 300
# dboxes = dboxes300_coco()
# train_trans = SSDTransformer(dboxes, (img_h, img_w), val=False)
# val_trans = SSDTransformer(dboxes, (img_h, img_w), val=True)



# local = '/media/mahdi/2e197b57-e3e6-4185-8d1b-5fbb1c3b8b55/datasets/modanet/'
def build_loaders(args):
    print("build loaders")
    path = args.modanet
    global dataset, testSet
    # dataset = COCODetection(img_folder=path + '/data', annotate_file=path + '/instances_train.json',
    #                         transform=train_trans)
    dataset = dset.CocoDetection(root=path + '/data', annFile=path + '/instances_train.json')
    dataset.transforms = torchvision.datasets.vision.StandardTransform(trans, annToTarget())

    testSet = dset.CocoDetection(root=path + '/data', annFile=path + '/instances_val.json')
    testSet.transforms = torchvision.datasets.vision.StandardTransform(trans, annToTarget())

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,num_workers=args.workers)
    test_loader = DataLoader(testSet, batch_size=3 * args.batch_size, shuffle=True, collate_fn=collate_fn,
                             num_workers=args.workers)
    return loader, test_loader


def train(args, model, device, loader, test_loader):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005, nesterov=True)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    num_epochs = args.epoch
    eval = evaluate(model, test_loader, device=device)
    print(eval)
    for epoch in range(args.start, args.epoch):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset

        if dist.is_primary():
            if args.distributed:
                torch.save(model.module.state_dict(), f"models/ssd_{str(epoch + 1).zfill(3)}.pt")
            else:
                torch.save(model.state_dict(), f"models/ssd_{str(epoch + 1).zfill(3)}.pt")
        # evaluate(model, test_loader, device=device)

    # torch.save(model.state_dict(), 'rcnn-last.pt')

def build_model():
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(num_classes=13)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    return model

def load_model(model, model_path):
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint, strict=True)
        print('==> loaded checkpoint {} (epoch)'
              .format(args.resume))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))

@torch.no_grad()
def testVis(imgPath, model, cats):
    model.eval()
    input = trans(Image.open(imgPath))
    input = input.unsqueeze(0).cuda()
    outputs = model(input)
    imgCV = cv2.imread(imgPath)
    for i in range(len(outputs[0]['boxes'])):
        item = outputs[0]
        s = item['scores'][i]
        if s < 0.5:
            continue
        m = item['masks'][i].expand(3, -1, -1).permute(1, 2, 0).cpu().numpy()
        l = item['labels'][i].item()
        b = item['boxes'][i]
        x, y, w, h = b.cpu().numpy().astype(np.int)
        # imgCV = (imgCV + (255 *  m).astype(np.uint8))
        cv2.rectangle(imgCV, (x, y), (w, h), (255, 0, 0), 2)
        label = cats[l+1]['name']
        cv2.putText(imgCV, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    cv2.imshow("a", imgCV)
    cv2.waitKey()

def main(args):
    print('main')
    os.makedirs('./models', exist_ok=True)
    args.distributed = dist.get_world_size() > 1
    model = build_model()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if len(args.resume) > 0:
        load_model(model, args.resume)

    loader, test_loader = build_loaders(args)

    # testVis("/home/mahdi/PycharmProjects/Datasets/SYSU-MM01/cam5/0015/0001.jpg", model,
    #         loader.dataset.coco.cats)
    #
    # return
    # loader, test_loader = build_loaders(args, '/media/mahdi/2e197b57-e3e6-4185-8d1b-5fbb1c3b8b55/datasets/modanet/')

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
        )
    train(args, model, device, loader=loader, test_loader=test_loader)


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
    parser.add_argument("--epoch", "-e", type=int, default=10)
    parser.add_argument("--start", "-s", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--sched", type=str)
    parser.add_argument("--batch_size", '-b', type=int, default=32)
    parser.add_argument('--resume', '-r', default='', type=str,
                        help='resume from checkpoint')
    parser.add_argument('--modanet', default='/export/livia/home/vision/malehdaghi/Datasets/modanet', type=str,
                        help='path to modanet')
    parser.add_argument('--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    args = parser.parse_args()
    torch.multiprocessing.set_sharing_strategy('file_system')
    print(args)

    dist.launch(main, args.n_gpu, 1, 0, args.dist_url, args=(args,))

    # main(args)




