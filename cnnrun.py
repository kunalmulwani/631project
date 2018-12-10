import torch
import torchvision.datasets as tdatasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from cnnmodel import Net
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch import optim
from FacesDataset import FacesDataset
def main():
    device = torch.device('cuda:9' if torch.cuda.is_available else 'cpu')
    train_dataset_dir = tdatasets.ImageFolder('images/all')
    train_dataset = FacesDataset(train_dataset_dir, \
                                 transform=transforms.Compose([transforms.Resize((100, 100)), transforms.ToTensor()]))

    train, test = split_train_val(train_dataset)
    vis_dataloader = DataLoader(train,
                                shuffle=True,
                                num_workers=2,
                                batch_size=10)
    model = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.75)

    loss_data = []
    for epoch in range(50):
        loss_val = 0.0
        for i, data in enumerate(vis_dataloader, 0):
            samples, labels = data
            labels = labels.squeeze()
            # print(labels.type())
            optimizer.zero_grad()
            outputs = model(samples)
            # print(labels)
            # print(outputs.shape, labels.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            loss_val  += loss.item()
            optimizer.step()
        loss_data.append(loss_val)

    plt.plot(loss_data)
    plt.savefig('loss.png')
    # plt.show()

    test_dataloader = DataLoader(test,
                                shuffle=True,
                                num_workers=2,
                                batch_size=10)

    total = 0
    correct = 0
    class_correct = [0 for i in range(40)]
    class_total = [0 for i in range(40)]
    for i, data in enumerate(test_dataloader, 0):
        samples, labels = data
        outputs = model(samples)
        print(outputs.data)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        labels = labels.squeeze()
        # print(labels)
        c = (predicted == labels).squeeze()
        # print(predicted.shape, labels.shape)
        print(predicted)
        correct += (predicted == labels).cpu().numpy().sum()



        for i in range(c.shape[0]): #mini_batch_size
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

    # print(class_total)
    for i in range(40):

        print('Accuracy of ', i, 100 * class_correct[i] / class_total[i])
        # accuracies.append((100 * class_correct[i] / class_total[i]))



    # for i, data in enumerate(vis_dataloader, 0):
    #     samples, labels, file_path = data
        # print(file_path[0])
        # print(samples[0].shape)
        # print(labels[0])
        # plt.imshow(samples[0].squeeze(0).numpy())
        # plt.show()

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


if __name__ == '__main__':
    main()
