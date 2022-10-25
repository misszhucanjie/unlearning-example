from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
import numpy as np
def dataloader(dataset, input_size, batch_size, split='train'):
    transform = transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    if dataset == 'mnist':
        data_loader = DataLoader(
            datasets.MNIST('data/mnist', train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'fashion-mnist':
        data_loader = DataLoader(
            datasets.FashionMNIST('data/fashion-mnist', train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'cifar10':
        data_loader = DataLoader(
            datasets.CIFAR10('data/cifar10', train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'noise-cifa10':
        unlearnable_train_dataset = datasets.CIFAR10('data/cifar10', train=True, download=True, transform=transform)
        noise = np.load('../Unlearnable-Examples-main/data_save/noise_posion.npy')
        noise = torch.from_numpy(noise)
        perturb_noise = noise.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu').numpy()
        unlearnable_train_dataset.data = unlearnable_train_dataset.data.astype(np.float32)
        label_index = [1, 2]
        for i in range(len(unlearnable_train_dataset)):
            # for index in range(len(unlearnable_train_dataset.data[i])):
            # if unlearnable_train_dataset.targets[i] in label_index:
            unlearnable_train_dataset.data[i] += perturb_noise[i]
            unlearnable_train_dataset.data[i] = np.clip(unlearnable_train_dataset.data[i], a_min=0, a_max=255)
        unlearnable_train_dataset.data = unlearnable_train_dataset.data.astype(np.uint8)
        data_loader = DataLoader(dataset=unlearnable_train_dataset,batch_size=batch_size, shuffle=True)
    elif dataset == 'noise-mnist':
        unlearnable_train_dataset = datasets.MNIST('data/mnist', train=True, download=True, transform=transform)
        noise = np.load('../Unlearnable-Examples-main/data_save/mnist_noise_posion.npy')
        noise = torch.from_numpy(noise)
        perturb_noise = noise.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu')
        unlearnable_train_dataset.data = unlearnable_train_dataset.data.to(dtype = torch.float)
        # unlearnable_train_dataset.data = np.array(unlearnable_train_dataset.data).astype(np.float32)
        label_index = [1, 2]
        for i in range(len(unlearnable_train_dataset)):
            for index in range(len(unlearnable_train_dataset.data[i])):
                if unlearnable_train_dataset.targets[i] in label_index:
                    unlearnable_train_dataset.data[i] += perturb_noise[i]
                    unlearnable_train_dataset.data[i] = np.clip(unlearnable_train_dataset.data[i], a_min=0, a_max=255)
        # unlearnable_train_dataset.data = unlearnable_train_dataset.data.astype(np.uint8)
        data_loader = DataLoader(dataset=unlearnable_train_dataset, batch_size=batch_size, shuffle=True)

    elif dataset == 'svhn':
        data_loader = DataLoader(
            datasets.SVHN('data/svhn', split=split, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)

    elif dataset == 'stl10':
        data_loader = DataLoader(
            datasets.STL10('data/stl10', split=split, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'lsun-bed':
        data_loader = DataLoader(
            datasets.LSUN('data/lsun', classes=['bedroom_train'], transform=transform),
            batch_size=batch_size, shuffle=True)

    return data_loader