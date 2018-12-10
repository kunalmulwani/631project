from torch.utils.data import Dataset
from PIL import Image
import torch
import random
import numpy as np
class FacesDataset(Dataset):
    def __init__(self, imageFolderDataset, transform=None):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform

    def __getitem__(self, index):
        # print(self.imageFolderDataset.imgs)
        data_tuple = self.imageFolderDataset.imgs[index]
        file_path_arr = data_tuple[0].split("/")
        label = int(file_path_arr[2][1:])
        # for simg in self.imageFolderDataset.imgs
        img = Image.open(data_tuple[0]).convert("L")
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.from_numpy(np.array([label-1],dtype=np.int64)) #, data_tuple[0]


    def __len__(self):
        return len(self.imageFolderDataset.imgs)