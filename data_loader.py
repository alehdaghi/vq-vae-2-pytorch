import os
from functools import reduce
from random import random

import numpy as np
import torch
from PIL import Image
import torch.utils.data as data
from torch.utils.data import Sampler
from torchvision.transforms import transforms
import pickle
import cv2
from utils.transforms import get_affine_transform, _box2cs

import torchvision.models.detection.mask_rcnn as mask_rcnn


class SYSUData(data.Dataset):
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            # transforms.CenterCrop(args.size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    def __init__(self, data_dir='../Datasets/SYSU-MM01/', transform=None, part=False):

        # data_dir = '../Datasets/SYSU-MM01/'
        # Load training images (path) and labels
        self.train_color_image = np.load(data_dir + 'train+Val_rgb_resized_img.npy')
        self.train_color_label = np.load(data_dir + 'train+Val_rgb_resized_label.npy')
        self.train_color_cam = np.load(data_dir + 'train+Val_rgb_resized_camera.npy')

        self.train_ir_image = np.load(data_dir + 'train+Val_ir_resized_img.npy')
        self.train_ir_label = np.load(data_dir + 'train+Val_ir_resized_label.npy')
        self.train_ir_cam = np.load(data_dir + 'train+Val_ir_resized_camera.npy')

        self.part = part
        if self.part:
            self.train_rgb_part = np.load(data_dir + 'train+Val_rgb_resized_part.npy')
            self.train_ir_part = np.load(data_dir + 'train+Val_ir_resized_part.npy')

        with open(data_dir + 'color_pos.pkl', 'rb') as f:
            self.color_pos = list(pickle.load(f).values())

        with open(data_dir + 'thermal_pos.pkl', 'rb') as f:
            self.thermal_pos = list(pickle.load(f).values())

        # self.color_pos, self.thermal_pos = GenIdx(self.train_color_label, self.train_ir_label)

        self.cIndex = np.arange(len(self.train_color_label))
        self.tIndex = np.arange(len(self.train_ir_label))
        # BGR to RGB

        self.transform = transform
        self.num_class = len(self.color_pos)

    def __getitem__(self, index):
        img1, target1, cam1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]], \
                              self.train_color_cam[self.cIndex[index]]
        img2, target2, cam2 = self.train_ir_image[self.tIndex[index]], self.train_ir_label[self.tIndex[index]], \
                              self.train_ir_cam[self.tIndex[index]]

        if self.part:
            parts1 = self.train_rgb_part[self.cIndex[index]]
            parts2 = self.train_ir_part[self.tIndex[index]]
            img1, parts1 = self.affine(img1, parts1)
            img2, parts2 = self.affine(img2, parts2)
            return self.transform(img1), self.transform(img2), target1, target2, cam1, cam2, parts1, parts2

        return self.transform(img1), self.transform(img2), target1, target2, cam1, cam2

    def __len__(self):
        return len(self.cIndex)

    def affine(self, img, partSeg, sf=0.25, rf=30):
        person_center, s = _box2cs([0, 0, img.shape[1] - 1, img.shape[0] - 1])
        s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
        r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) if random() <= 0.6 else 0

        trans = get_affine_transform(person_center, s, r, img.shape[:2])
        img2 = cv2.warpAffine(
            img,
            trans,
            (int(img.shape[1]), int(img.shape[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))
        partSeg2 = cv2.warpAffine(
            partSeg,
            trans,
            (int(img.shape[1]), int(img.shape[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255))
        return img2, partSeg2

    @staticmethod
    def rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(rgb.dtype)

    @staticmethod
    def rgb2RandomChannel(rgb):
        n = np.random.rand(3)
        n /= n.sum()
        return np.dot(rgb[..., :3], n).astype(rgb.dtype)

    def samplize(self, batch_size, num_pos):
        sampler = IdentitySampler(self.train_color_label,
                                            self.train_ir_label, self.color_pos, self.thermal_pos, num_pos,
                                            batch_size)

        self.cIndex = sampler.index1  # color index
        self.tIndex = sampler.index2
        return sampler


class IdentitySampler(Sampler):
    """Sample person identities evenly in each batch.
        Args:
            train_color_label, train_ir_label: labels of two modalities
            color_pos, thermal_pos: positions of each identity
            batchSize: batch size
    """

    def __init__(self, train_color_label, train_ir_label, color_pos, thermal_pos, num_pos, batchSize):
        uni_label = np.unique(train_color_label)
        self.n_classes = len(uni_label)

        N = np.maximum(len(train_color_label), len(train_ir_label))
        for j in range(int(N / (batchSize * num_pos)) + 1):
            batch_idx = np.random.choice(uni_label, batchSize, replace=False)
            for i in range(batchSize):
                sample_color = np.random.choice(color_pos[batch_idx[i]], num_pos)
                sample_thermal = np.random.choice(thermal_pos[batch_idx[i]], num_pos)

                if j == 0 and i == 0:
                    index1 = sample_color
                    index2 = sample_thermal
                else:
                    index1 = np.hstack((index1, sample_color))
                    index2 = np.hstack((index2, sample_thermal))

        self.index1 = index1
        self.index2 = index2
        self.N = N

    def __iter__(self):
        return iter(np.arange(len(self.index1)))

    def __len__(self):
        return self.N


def GenIdx(train_color_label, train_ir_label):
    color_pos = {}
    unique_label_color = np.unique(train_color_label)
    for i in range(len(unique_label_color)):
        tmp_pos = [k for k, v in enumerate(train_color_label) if v == unique_label_color[i]]
        color_pos[i] = tmp_pos

    f = open("color_pos.pkl", "wb")
    pickle.dump(color_pos, f)

    thermal_pos = {}
    unique_label_thermal = np.unique(train_ir_label)
    for i in range(len(unique_label_thermal)):
        tmp_pos = [k for k, v in enumerate(train_ir_label) if v == unique_label_thermal[i]]
        thermal_pos[i] = tmp_pos

    f = open("thermal_pos.pkl", "wb")
    pickle.dump(thermal_pos, f)

    return list(color_pos.values()), list(thermal_pos.values())


class TestData(data.Dataset):
    def __init__(self, test_img_file, test_label, test_cam, transform=None, img_size = (144,288), colorToGray=False):
        self.gray = colorToGray
        test_image = []
        ret_test_label,  ret_test_cam = [],[]
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            if colorToGray:
                #for j in range(9):
                pix_array = np.stack((SYSUData.rgb2gray(pix_array),)*3, axis=-1)
                # ret_test_label.append(test_label[i])
                # ret_test_cam.append(test_cam[i])

            ret_test_cam.append(test_cam[i])
            ret_test_label.append(test_label[i])
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = np.array(ret_test_label)
        self.test_cam = np.array(ret_test_cam)
        self.transform = transform

    def __getitem__(self, index):
        img1,  target1, cam1 = self.test_image[index],  self.test_label[index], self.test_cam[index]
        img1 = self.transform(img1)
        return img1, target1, cam1 - 1

    def __len__(self):
        return len(self.test_image)


def process_sysu(data_path, data='query', single_shot=True, mode ='all', file_path='exp/test_id.txt'):
    cameras = []
    if data == 'query':
        if mode == 'all':
            cameras = ['cam3','cam6']
        elif mode =='indoor':
            cameras = ['cam3','cam6']
        elif mode == 'Vis' or mode == 'Gray': #test unimodal training
            cameras = ['cam1', 'cam4']
        elif mode == 'Ir':
            cameras = ['cam3']
    else:
        if mode == 'all':
            cameras = ['cam1', 'cam2', 'cam4', 'cam5']
        elif mode == 'indoor':
            cameras = ['cam1', 'cam2']
        elif mode == 'Vis' or mode == 'Gray':  # test unimodal training
            cameras = ['cam2', 'cam5']
        elif mode == 'Ir':
            cameras = ['cam6']

    file_path = os.path.join(data_path,file_path)
    files = []


    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    for id in sorted(ids):
        for cam in cameras:
            img_dir = os.path.join(data_path,cam,id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
                if data == 'gallery' and single_shot:
                    files.append(np.random.choice(new_files))
                else:
                    # files_rgb.extend(random.choices(new_files, k = 10))
                    files.extend(new_files)
    imgs = []
    ids = []
    cams = []
    for img_path in files:
        camid, pid = int(img_path[-15]), int(img_path[-13:-9])
        imgs.append(img_path)
        ids.append(pid)
        cams.append(camid)
    return imgs, np.array(ids), np.array(cams)
