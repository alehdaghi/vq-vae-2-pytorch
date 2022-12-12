import argparse
import os
from pycocotools.coco import COCO
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.models.detection.mask_rcnn as mask_rcnn
from vision.engine import *
import distributed as dist

img_h, img_w = 288, 144

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
trans = transforms.Compose([
    transforms.ToTensor(),
#     normalize,
])
coco = None

def annToTarget(ann):
    N = len(ann)
    boxes = torch.FloatTensor(N, 4)
    labels = torch.LongTensor(N)
    image_id = torch.LongTensor(N)
    area = torch.IntTensor(N)
    iscrowd = torch.IntTensor(N)
    masks = []#torch.empty((N, 600, 400), dtype=torch.uint8)
    for i,obj in enumerate(ann):
        boxes[i] = torch.FloatTensor(obj['bbox'])
        labels[i] = obj['category_id']
        image_id[i] = obj['image_id']
        area[i] = obj['area']
        iscrowd[i] = obj['iscrowd']
        masks.append(torch.from_numpy(coco.annToMask(obj)))

    boxes[:, 2:] += boxes[:, :2]

    dict = {
        'boxes' : boxes,
        'labels': labels - 1,
        'image_id': image_id,
        "area" : area,
        "iscrowd" : iscrowd,
        "masks" : torch.stack(masks) if N > 0 else torch.empty((N, 300, 200))

    }
    return dict

def collate_fn(batch):
    return tuple(zip(*batch))

#local = '/media/mahdi/2e197b57-e3e6-4185-8d1b-5fbb1c3b8b55/datasets/modanet/'
def build_loaders(args):
    path = args.modanet
    global dataset, testSet
    dataset = dset.CocoDetection(root=path + '/images', annFile = path + '/annotations/modanet2018_instances_train.json',
                                    transform=trans, target_transform=annToTarget)
    testSet = dset.CocoDetection(root = path + '/images', annFile=path +'/annotations/modanet2018_instances_val.json',
                                    transform=trans, target_transform=annToTarget)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.workers)
    test_loader = DataLoader(testSet, batch_size=4 * args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.workers)
    return loader, test_loader



def train(args, model, device, loader, test_loader):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,  momentum=0.9, weight_decay=0.0005, nesterov=True)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = args.epoch
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, test_loader, device=device)
        if dist.is_primary():
            torch.save(model.state_dict(), f"rcn/rcn_{str(epoch + 1).zfill(3)}.pt")

    # torch.save(model.state_dict(), 'rcnn-last.pt')


def main(args):
    args.distributed = dist.get_world_size() > 1
    grcnn = torchvision.models.detection.transform.GeneralizedRCNNTransform(min_size=200, max_size=300,
                                                                            image_mean=[0.485, 0.456, 0.406],
                                                                            image_std=[0.229, 0.224, 0.225])
    model = mask_rcnn.maskrcnn_resnet50_fpn_v2(mask_rcnn.MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=13)
    model.roi_heads.mask_predictor = mask_rcnn.MaskRCNNPredictor(256, 256, 13)
    model.transform = grcnn
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

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

    # loader, test_loader = build_loaders(args, '/media/mahdi/2e197b57-e3e6-4185-8d1b-5fbb1c3b8b55/datasets/modanet/')
    loader, test_loader = build_loaders(args)
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
    parser.add_argument('--modanet' , default='/export/livia/home/vision/malehdaghi/Datasets/modanet', type=str,
                        help='path to modanet')
    parser.add_argument('--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    args = parser.parse_args()
    torch.multiprocessing.set_sharing_strategy('file_system')
    print(args)
    
    coco = COCO(args.modanet + '/annotations/modanet2018_instances_train.json')
    dist.launch(main, args.n_gpu, 1, 0, args.dist_url, args=(args,))


    # main(args)
