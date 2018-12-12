from torch.utils.data import Dataset
import random
from PIL import Image
import torch
import numpy as np
import torchvision.datasets as tdatasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class SiameseNetworkDataset(Dataset):
    
    def __init__(self,imageFolderDataset,transform=None):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        self.labelled_data = [[] for i in range(40)]

        self.data = [[] for i in range(40)]
        for i in range(len(self.imageFolderDataset.imgs)):
            img_tpl = self.imageFolderDataset.imgs[i]
            label = self.get_image_label(img_tpl)
            self.labelled_data[label].append(img_tpl[0])
        for i in range(40):
            for j in range(1,8):
                self.data[i].append((self.labelled_data[i][0],self.labelled_data[i][j]))
            for j in range(7):
                rand_num = random.randint(0,39)
                while rand_num == i:
                    rand_num = random.randint(0, 39)
                self.data[i].append((self.labelled_data[i][0], self.labelled_data[rand_num][1]))

        
    def __getitem__(self,index):
        # print('data idx', index//2)
        # print(len(self.data), len(self.data[index//2]))
        img0 = self.data[index//14][index%14][0]
        img1 = self.data[index//14][index%14][1]
        img0 = Image.open(img0).convert("L")
        img1 = Image.open(img1).convert("L")
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        index = index%14
        if index<7:
            index=0
        else:
            index=1
        return img0, img1, torch.from_numpy(np.array([index],dtype=np.float32))

        # return img0, img1 , torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32))
    
    def __len__(self):
        return len(self.data)*len(self.data[0]) #len(self.imageFolderDataset.imgs)

    def get_image_label(self, data_tuple):
        # data_tuple = self.imageFolderDataset.imgs[idx]
        file_path_arr = data_tuple[0].split("/")
        label = int(file_path_arr[2][1:])
        return label-1
        # img = Image.open(data_tuple[0]).convert("L")
        # if self.transform is not None:
        #     img = self.transform(img)
        # img, torch.from_numpy(np.array([label-1],dtype=np.int64))

    def get_image(self, data_tuple):
        img = Image.open(data_tuple[0]).convert("L")
        if self.transform is not None:
            img = self.transform(img)
        return img

if __name__ == "__main__":
    train_dataset_dir = tdatasets.ImageFolder('images/all')
    train = SiameseNetworkDataset(imageFolderDataset = train_dataset_dir, transform = transforms.Compose([transforms.Resize((100,100)), transforms.ToTensor()]))
    dataloader = DataLoader(train, shuffle=False, num_workers=0, batch_size=1)
    for i,data in enumerate(dataloader, 0):
        img_0, img_1, label = data
        print(i, label)
        if i == 28:
            break

