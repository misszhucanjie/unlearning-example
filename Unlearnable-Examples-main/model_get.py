import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import  os
import numpy as np
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# Prepare Dataset
train_transform = [
    transforms.ToTensor()
]
test_transform = [
    transforms.ToTensor()
]
train_transform = transforms.Compose(train_transform)
test_transform = transforms.Compose(test_transform)

clean_train_dataset = datasets.CIFAR10(root='../datasets', train=False, download=True, transform=train_transform)
# clean_train_dataset = datasets.CIFAR10(root='../datasets', train=False, download=True, transform=test_transform)
clean_test_dataset = datasets.CIFAR10(root='../datasets', train=False, download=True, transform=test_transform)

clean_train_loader = DataLoader(dataset=clean_train_dataset, batch_size=512,
                                shuffle=False, pin_memory=True,
                                drop_last=False, num_workers=12)
clean_test_loader = DataLoader(dataset=clean_test_dataset, batch_size=512,
                                shuffle=False, pin_memory=True,
                                drop_last=False, num_workers=12)

from models.ResNet import ResNet18
import toolbox

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

base_model = ResNet18()


path = './point_best_module.pkl'
base_model = torch.load(path)

noise = np
base_model = torch.nn.DataParallel(base_model)
base_model = base_model.cuda()
base_model.eval()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=base_model.parameters(), lr=0.1, weight_decay=0.0005, momentum=0.9)

noise_generator = toolbox.PerturbationTool(epsilon=0.03137254901960784, num_steps=20, step_size=0.0031372549019607846)

from tqdm import tqdm

# noise = torch.zeros([50000, 3, 32, 32])
noise = np.load('./data_save/noise_posion.npy')
noise = torch.from_numpy(noise)
data_iter = iter(clean_train_loader)
condition = True
train_idx = 0


epochs = 1


import numpy as np

# Add standard augmentation
train_transform = [
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
]
train_transform = transforms.Compose(train_transform)
clean_train_dataset = datasets.CIFAR10(root='../datasets', train=True, download=True, transform=train_transform)
unlearnable_train_dataset = datasets.CIFAR10(root='../datasets', train=True, download=True, transform=train_transform)

perturb_noise = noise.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu').numpy()
unlearnable_train_dataset.data = unlearnable_train_dataset.data.astype(np.float32)
label_index = [1,2]
for i in range(len(unlearnable_train_dataset)):
    # for index in range(len(unlearnable_train_dataset.data[i])):
    if unlearnable_train_dataset.targets[i] in label_index:
        unlearnable_train_dataset.data[i] += perturb_noise[i]
        unlearnable_train_dataset.data[i] = np.clip(unlearnable_train_dataset.data[i] , a_min=0, a_max=255)
unlearnable_train_dataset.data = unlearnable_train_dataset.data.astype(np.uint8)

import random
import matplotlib.pyplot as plt
import matplotlib
# % matplotlib
# inline


def imshow(img):
    fig = plt.figure(figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def get_pairs_of_imgs(idx):
    clean_img = clean_train_dataset.data[idx]
    unlearnable_img = unlearnable_train_dataset.data[idx]
    clean_img = torchvision.transforms.functional.to_tensor(clean_img)
    unlearnable_img = torchvision.transforms.functional.to_tensor(unlearnable_img)

    x = noise[idx]
    x_min = torch.min(x)
    x_max = torch.max(x)
    noise_norm = (x - x_min) / (x_max - x_min)
    noise_norm = torch.clamp(noise_norm, 0, 1)
    return [clean_img, noise_norm, unlearnable_img]


selected_idx = [random.randint(0, 50000) for _ in range(3)]
img_grid = []
img_grid_ori = []
img_grid_noise = []
img_grid_adv = []
for idx in selected_idx:
    img_grid += get_pairs_of_imgs(idx)
    img_grid_ori += [get_pairs_of_imgs(idx)[0]]
    img_grid_noise += [get_pairs_of_imgs(idx)[1]]
    img_grid_adv += [get_pairs_of_imgs(idx)[2]]


imshow(torchvision.utils.make_grid(torch.stack(img_grid), nrow=3, pad_value=255))
imshow(torchvision.utils.make_grid(torch.stack(img_grid_ori), nrow=1, pad_value=255))
imshow(torchvision.utils.make_grid(torch.stack(img_grid_noise), nrow=1, pad_value=255))
imshow(torchvision.utils.make_grid(torch.stack(img_grid_adv), nrow=1, pad_value=255))

from util import AverageMeter

model = ResNet18()
model = model.cuda()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1, weight_decay=0.0005, momentum=0.9)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=0)

unlearnable_loader = DataLoader(dataset=unlearnable_train_dataset, batch_size=128,
                                shuffle=True, pin_memory=True,
                                drop_last=False, num_workers=12)
label_list = [[] for i in range(10)]
acc_noise_ori = [[] for i in range(10)]
for epoch in range(30):
    # Train
    model.train()
    acc_meter = AverageMeter()
    loss_meter = AverageMeter()
    pbar = tqdm(unlearnable_loader, total=len(unlearnable_loader))
    for images, labels in pbar:
        images, labels = images.cuda(), labels.cuda()
        model.zero_grad()
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        _, predicted = torch.max(logits.data, 1)
        acc = (predicted == labels).sum().item() / labels.size(0)
        for i in range(len(predicted)):
            if predicted[i] == labels[i]:
                acc_noise_ori[labels[i]].append(1)
            else:
                acc_noise_ori[labels[i]].append(0)

        acc_meter.update(acc)
        loss_meter.update(loss.item())
        pbar.set_description("Acc %.2f Loss: %.2f" % (acc_meter.avg * 100, loss_meter.avg))
    scheduler.step()

    for i in range(len(acc_noise_ori)):
        correct_list = sum(acc_noise_ori[i])/len(acc_noise_ori[i])
        print('clean  index{}correct_list{}'.format(i,correct_list))

    # Eval
    model.eval()
    correct, total = 0, 0
    for i, (images, labels) in enumerate(clean_test_loader):
        images, labels = images.cuda(), labels.cuda()
        with torch.no_grad():
            logits = model(images)
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            for i in range(len(predicted)):
                if predicted[i] == labels[i]:
                    label_list[labels[i]].append(1)
                else:
                    label_list[labels[i]].append(0)
            correct += (predicted == labels).sum().item()
    acc = correct / total
    tqdm.write('Clean Accuracy %.2f\n' % (acc * 100))
    for i in range(len(label_list)):
        correct_list = sum(label_list[i])/len(label_list[i])
        print('clean  index{}correct_list{}'.format(i,correct_list))
