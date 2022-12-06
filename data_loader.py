from functools import reduce

import numpy as np
import torch
from PIL import Image
import torch.utils.data as data
from torch.utils.data import Sampler
from torchvision.transforms import transforms
import pickle


class SYSUData(data.Dataset):
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            # transforms.CenterCrop(args.size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    def __init__(self, data_dir='../Datasets/SYSU-MM01/', transform=None):

        # data_dir = '../Datasets/SYSU-MM01/'
        # Load training images (path) and labels
        self.train_color_image = np.load(data_dir + 'train+Val_rgb_resized_img.npy')
        self.train_color_label = np.load(data_dir + 'train+Val_rgb_resized_label.npy')
        self.train_color_cam = np.load(data_dir + 'train+Val_rgb_resized_camera.npy')

        self.train_ir_image = np.load(data_dir + 'train+Val_ir_resized_img.npy')
        self.train_ir_label = np.load(data_dir + 'train+Val_ir_resized_label.npy')
        self.train_ir_cam = np.load(data_dir + 'train+Val_ir_resized_camera.npy')

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
        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, target1, target2, cam1, cam2

    def __len__(self):
        return len(self.train_color_image) + len(self.train_ir_image)

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
