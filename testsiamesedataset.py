from torch.utils.data import Dataset
import random
from PIL import Image
import torch
import numpy as np


class SiameseTestDataset(Dataset):

    def __init__(self, imageFolderDataset, transform=None):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.labelled_data = [[] for i in range(40)]

        self.data = [None for i in range(40)]
        for i in range(len(self.imageFolderDataset.imgs)):
            img_tpl = self.imageFolderDataset.imgs[i]
            label = self.get_image_label(img_tpl)
            self.labelled_data[label].append(img_tpl[0])
        for i in range(40):
            self.data[i] = self.labelled_data[i][0]

    def __getitem__(self, index):
        # print('data idx', index//2)
        # print(len(self.data), len(self.data[index//2]))
        img0 = self.data[index]
        # img1 = self.data[index // 2][index % 2][1]
        img0 = Image.open(img0).convert("L")
        # img1 = Image.open(img1).convert("L")
        if self.transform is not None:
            img0 = self.transform(img0)
            # img1 = self.transform(img1)
        return img0

        # return img0, img1 , torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32))

    def __len__(self):
        return 40  # len(self.imageFolderDataset.imgs)

    def get_image_label(self, data_tuple):
        # data_tuple = self.imageFolderDataset.imgs[idx]
        file_path_arr = data_tuple[0].split("/")
        label = int(file_path_arr[2][1:])
        return label - 1
        # img = Image.open(data_tuple[0]).convert("L")
        # if self.transform is not None:
        #     img = self.transform(img)
        # img, torch.from_numpy(np.array([label-1],dtype=np.int64))

    def get_image(self, data_tuple):
        img = Image.open(data_tuple[0]).convert("L")
        if self.transform is not None:
            img = self.transform(img)
        return img