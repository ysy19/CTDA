#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
import torch.optim.lr_scheduler
import torchvision
from torchvision import datasets, transforms

from tqdm.notebook import tqdm as tqdm


# In[ ]:
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--gpu',required=True,type=str,default='0,1,2,3', help='gpu 설정')
args = parser.parse_args()

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu


if torch.cuda.is_available() :
    print('avail')
else: 
    print('cpu')


# **Model Define**

# In[ ]:


from torchvision import models
import torch




resnet101 = models.resnet101(pretrained=True)


# **Utils**

# In[ ]:


class AverageMeter(object):
    r"""Computes and stores the average and current value
    """
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    r"""Computes the accuracy over the $k$ top predictions for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # _, pred = output.topk(maxk, 1, True, True)
        # pred = pred.t()
        # correct = pred.eq(target.view(1, -1).expand_as(pred))

        # faster topk (ref: https://github.com/pytorch/pytorch/issues/22812)
        _, idx = output.sort(descending=True)
        pred = idx[:,:maxk]
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# **CutOut**

# In[ ]:


class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


from torchvision.datasets import ImageFolder 

class CustomDataset(torchvision.datasets.ImageFolder):
    def __getitem__(self, index): 
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path
# **Parameter Setting**

# In[ ]:


dataset = 'imagenet' # cifar10 or cifar100
model = 'Resnet101' # resnet18, resnet50, resnet101
batch_size = 128  # Input batch size for training (default: 128)
epochs = 20 # Number of epochs to train (default: 200)
learning_rate = 0.1 # Learning rate
data_augmentation = False # Traditional data augmentation such as augmantation by flipping and cropping?
cutout = False # Apply Cutout?
n_holes = 1 # Number of holes to cut out from image
length = 16 # Length of the holes
seed = 0 # Random seed (default: 0)
print_freq = 500
cuda = torch.cuda.is_available()
cudnn.benchmark = True  # Should make training should go faster for large models

torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

test_id = dataset + '_' + model


# **Load and preprocess data**

# In[ ]:


from torchvision.datasets import ImageFolder 

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Normalize((0.5), (0.5)),
    transforms.transforms.RandomHorizontalFlip(p=0.5),
    transforms.transforms.RandomVerticalFlip(p=0.5),
])

#root1 = '/home/dircon/data/ILSVRC/Data/CLS-LOC/train/'
root1 = '### Imagenet train 데이터 위치###'
root2 = '### Imagenet test 데이터 위치###'

#trainset = ImageFolder(root=root1, transform=transform )
trainset=CustomDataset(root=root1, transform=transform)
#testset = ImageFolder(root=root2, transform=transform )
testset=CustomDataset(root=root2, transform=transform)


# In[ ]:


trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=4)


# **Train**

# In[ ]:
import random
from PIL import Image
import cv2

def train(train_loader, epoch, model, optimizer, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, losses,
                             top1, top5, prefix="Epoch: [{}]".format(epoch))
    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target,root) in enumerate(train_loader):
        new_input=input
        for j in range(0,len(root)):
            transparency_list=[0,10,20,30,40,50,60,70,80,90,100]
            transparency=transparency_list[random.randint(0,10)]
            root_j=root[j]
            if transparency==0:
                root_get='###Imagenet train 데이터 위치###'+root_j.split('/')[-2]+'/'+root_j.split('/')[-1]
                new_img=Image.open(root_get)
            elif transparency==100:
                root_get='###Imagenet 배경제거이미지 위치###'+root_j.split('/')[-2]+'/'+root_j.split('/')[-1]
                new_img=Image.open(root_get)
            else:
                root_img=root_j
                root_mask='###Imagenet 배경제거이미지 위치###'+root_j.split('/')[-2]+'/'+root_j.split('/')[-1]
                img=cv2.imread(root_img)
                mask=mask=cv2.imread(root_mask)
                mask=cv2.resize(mask, (img.shape[1], img.shape[0]))
                masked_transparency=cv2.addWeighted(img,1-(transparency/100),mask,transparency/100,0)
                #cv2.imwrite('/home/dircon/seoyun/tempImg.JPEG',masked_transparency)
                #root_get='/home/dircon/seoyun/tempImg.JPEG'
                #new_img=Image.open(root_get)
                new_img=Image.fromarray(masked_transparency)

            new_img=transform(new_img)
            if new_input[j].shape != new_img.shape:
                new_input[j]=input[j]
            else:
                new_input[j]=new_img
            #new_input[j]=new_img
        

        # measure data loading time
        #input = input.cuda()
        input = new_input.cuda()
        target = target.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss, accuracy 
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0].item(), input.size(0))
        top5.update(acc5[0].item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.print(i)

    print('==> Train Accuracy: Acc@1 {top1.avg:.3f} || Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    return top1.avg,top5.avg

def test(test_loader,epoch, model):
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    model.eval()
    for i,(input,target,root) in enumerate(test_loader):
        input = input.cuda()
        target = target.cuda()

        output = model(input)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0].item(), input.size(0))
        top5.update(acc5[0].item(), input.size(0))
    
    print('==> Test Accuracy:  Acc@1 {top1.avg:.3f} || Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    return top1.avg,top5.avg

if torch.cuda.is_available() :
    print('avail')
else: 
    print('cpu')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = nn.DataParallel(resnet101).to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, nesterov=True, weight_decay=5e-4)

#scheduler = MultiStepLR(optimizer, milestones=[60, 90, 120], gamma=0.2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = 50, eta_min = 1e-6)

criterion = torch.nn.CrossEntropyLoss().cuda()
###########################################################
best_acc = 0

#log를 파일로 저장하기
logs=[]

for epoch in range(epochs):
    print("\n----- epoch: {}, lr: {} -----".format(
        epoch, optimizer.param_groups[0]["lr"]))

    # train for one epoch
    start_time = time.time()
    train1,train2=train(trainloader, epoch, model, optimizer, criterion)
    test_acc,test_acc2 = test(testloader,epoch,model)
    
    line=[train1,train2,test_acc,test_acc2]
    logs.append(line)
    

    elapsed_time = time.time() - start_time
    print('==> {:.2f} seconds to train this epoch\n'.format(elapsed_time))
    # learning rate scheduling
    scheduler.step()
    
    # Save model for best accuracy
    if best_acc < test_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), 'model_best_color_white.pt')

    torch.save(model.state_dict(),'model_latest_color_white.pt')
    print(f"Best Top-1 Accuracy: {best_acc}")


import csv
with open('###저장 위치###/train_color_white.csv', 'w',newline='') as f: 
    write = csv.writer(f) 
      
    write.writerow(['train top1','train top5','test top1','test top5'])
    write.writerows(logs) 
        
