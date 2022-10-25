import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

# Prepare Dataset
train_transform = [
    transforms.ToTensor()
]
test_transform = [
    transforms.ToTensor()
]
train_transform = transforms.Compose(train_transform)
test_transform = transforms.Compose(test_transform)

clean_train_dataset = datasets.CIFAR10(root='../datasets', train=True, download=True, transform=train_transform)
clean_test_dataset = datasets.CIFAR10(root='../datasets', train=False, download=True, transform=test_transform)

clean_train_loader = DataLoader(dataset=clean_train_dataset, batch_size=512,
                                shuffle=False, pin_memory=True,
                                drop_last=False, num_workers=12)
clean_test_loader = DataLoader(dataset=clean_test_dataset, batch_size=512,
                                shuffle=False, pin_memory=True,
                                drop_last=False, num_workers=12)
class_data_save = [[] for i in range(10)]
for data,label in clean_train_loader:
    # print(data.shape)
    # print(label)
    for index in range(len(label)):
        class_data_save[label[index]].append(np.array(data[index]))
class_data_save = np.array(class_data_save)
print(class_data_save)
np.save('./data_save/data_inclede',class_data_save)
