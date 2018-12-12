import torchvision
import torchvision.datasets as tdatasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
#from dataset import SiameseNetworkDataset
from blog_dataset import SiameseNetworkDataset
from FacesDataset import FacesDataset
from testsiamesedataset import SiameseTestDataset
from model import SiameseNetwork, ContrastiveLoss
from torch import optim
import torch.nn.functional as F
import torch.nn as nn

def imshow(img,name,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if should_save:
        plt.savefig('images/'+ str(name) +'.png')
    else:
        plt.show()

def split_train_val(train):
    examples = {}
    for example in train:
        data, label = example
        if label.item() not in examples:
            examples[label.item()] = []
        examples[label.item()].append(example)

    validation_size = 0.2
    # for label in examples:
    #     print(label, len(examples[label]))
    train_80 = []
    valid_20 = []
    for label in examples:
        sample_size = int(np.floor(0.2 * len(examples[label])))
        choices = np.random.choice(len(examples[label]), sample_size, replace=False)
        # print(len(choices), len(examples[label]), sample_size)
        for choice in choices:
            valid_20.append(examples[label][choice])
        for i in range(len(examples[label])):
            if not i in choices:
                train_80.append(examples[label][i])



    return train_80, valid_20


def main():
    device = torch.device('cuda:9' if torch.cuda.is_available else 'cpu')
    train_dataset_dir = tdatasets.ImageFolder('images/train')
    train_dataset = SiameseNetworkDataset(imageFolderDataset = train_dataset_dir, transform = transforms.Compose([transforms.Resize((100,100)), transforms.ToTensor()]))
    vis_dataloader = DataLoader(train_dataset,
                        shuffle=False,
                        num_workers=0,
                        batch_size=1)

    # dataiter = iter(vis_dataloader)
    # example_batch = next(dataiter)
    # concatenated = torch.cat((example_batch[0],example_batch[1]),0)
    # imshow(torchvision.utils.make_grid(concatenated))
    # print(example_batch[2].numpy())
    #
    # example_batch = next(dataiter)
    # concatenated = torch.cat((example_batch[0], example_batch[1]), 0)
    # imshow(torchvision.utils.make_grid(concatenated))
    # print(example_batch[2].numpy())
    net = SiameseNetwork()
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(net.parameters(),lr = 0.0005 )
    loss_vals = []
    net.to(device)
    '''
    Training Starts
    '''
    print('Training started')
    for epoch in range(1000):
       loss_epoch = 0
       for i, data in enumerate(vis_dataloader,0):
           img_0, img_1, label = data
           img_0, img_1, label = img_0.to(device), img_1.to(device), label.to(device)
           optimizer.zero_grad()
           out_0, out_1 = net(img_0, img_1)
           loss = criterion(out_0, out_1, label)
           loss_epoch += loss.item()
           loss.backward()
           optimizer.step()
       loss_vals.append(loss_epoch)
       print('Epoch',str(epoch+1), str(loss_epoch))
    print('Training completed')
    plt.plot(loss_vals)
    plt.savefig('loss_siamese.png')
    
    

    # ****************************** Training ends ***************************************


    '''
    Testing starts
    '''
    

    test_dataset_dir = tdatasets.ImageFolder('images/test')
    net.load_state_dict(torch.load('siamese.pt'))
    test_dataset = SiameseNetworkDataset(imageFolderDataset = test_dataset_dir, transform = transforms.Compose([transforms.Resize((100,100)), transforms.ToTensor()]))

    test_dataloader = DataLoader(test_dataset,
                        shuffle=True,
                        num_workers=2,
                        batch_size=1)
    print('Testing starts')
    correct = 0
    total = 0
    test_img_sub = None
    for i, data in enumerate(test_dataloader, 0):
        img_0, img_1, label = data
        if test_img_sub is None:
            test_img_sub = img_0
        #concat = torch.cat((test_img_sub, img_1), 0)
        concat = torch.cat((img_0, img_1), 0)
        test_img_sub, img_1, label = test_img_sub.to(device), img_1.to(device), label.to(device)
        img_0, img_1, label = img_0.to(device), img_1.to(device), label.to(device)
        out_0, out_1 = net(img_0, img_1)
        dist = F.pairwise_distance(out_0, out_1)
        if dist <= 0.5 and label == 0:
            correct = correct + 1
        elif label == 1:
            correct = correct + 1
        else:
            pass
        total = total + 1
        imshow(torchvision.utils.make_grid(concat),i,'Dissimilarity: {:.2f}'.format(dist.item()), True)
        test_img_sub = test_img_sub.cpu()
#        dist = dist.cpu().detach()
#        print(dist.numpy())
#        dist = torch.sigmoid(dist)
#        print(dist.numpy())
    print(correct/total)


    print('Testing complete')

    torch.save(net.state_dict(), 'siamese_blog.pt')

    # example_batch = next(dataiter)
    # concatenated = torch.cat((example_batch[0],example_batch[1]),0)
    # imshow(torchvision.utils.make_grid(concatenated))
    # print(example_batch[2].numpy())
    

if __name__ == "__main__":
    main()
