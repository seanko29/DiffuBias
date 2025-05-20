"""
2023-09-01
Part1: Training biased classfiers & Extracting bias-conflict samples with methodology
This implementation is for understaning the overall structure of our methods.
"""

import argparse
import os, cv2, json, random
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from PIL import Image, ImageDraw
import sklearn.metrics as m

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import torchvision
import torchvision.transforms as T

from models import * # calling call_by_name models = mlp, conv ... and others...
from utils import *    # Generalized Cross Entropy
from adamp import AdamP

# Pytorch determistic
random_seed = 1
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)


class CustomMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, res):
        self.dict = res

    def update(self, key, score):        
        self.dict[key].append(score)

    def save(self, target_directory, filename):
        if filename:
            pd.DataFrame(self.dict, index=None).to_csv(f'{target_directory}/{filename}.csv')
        else:
            pd.DataFrame(self.dict, index=None).to_csv(target_directory+'results.csv')

    def is_best(self, key):
        if len(self.dict[list(self.dict.keys())[0]]) == 1:
            return True
        maximum = max(self.dict[key][:-1])
        current = self.dict[key][-1]

        if maximum < current:
            return True

    def print_info(self, key):
        best1   = round(max(self.dict[key]),4)
        current1= round(self.dict[key][-1], 4)

        str1 = f'Best/Curr {key}: {best1}/{current1}'
        print(f'\t{str1}')

def save(state, epoch, save_dir, model, is_parallel=None):
    os.makedirs(save_dir, exist_ok=True)
    
    target_path = f'{save_dir}/{state}.path.tar'
    
    with open(target_path, "wb") as f:
        if not is_parallel:
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),}, f)
        else:
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),}, f)

import torchvision.transforms as T

def get_transform(args):
    
    train_transform = T.Compose([
        T.Resize(args.image_size),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        # T.ColorJitter(),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Replace with actual mean and std values
    ])

    valid_transform = T.Compose([
        T.Resize(args.image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Replace with actual mean and std values
    ])

    return train_transform, valid_transform

class BaseDataset(nn.Module):
    def __init__(self, path, args, transform=None, mode='train'):
        super(BaseDataset, self).__init__()
        self.path = path
        self.args = args
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        origin_path = self.path[idx]
        # print(origin_path, 'path')
        # print(origin_path.split('.')[0].split('/')[-2], 'check label')
        
        if self.mode == 'valid-bffhq':
            label = int(origin_path.split('.')[0].split('_')[-2])
        elif self.mode == 'valid':
            
            label = int(origin_path.split('_')[-2])
        else:
            # print(origin_path)
            if 'img' in origin_path: # for all shits generated
                label = int(origin_path.split('/')[-4])
            else:
                if '0.5pct' in origin_path:
                    label = int(origin_path.split('.')[1].split('/')[-2])
                else:
                    label = int(origin_path.split('.')[0].split('/')[-2])
            
            # print(label, 'check label')
        image = Image.open(origin_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return idx, image, label, origin_path




def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    for milestone in args.lr_decay_schedule:
        lr *= args.lr_decay_rate if epoch >= milestone else 1.

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def set_args(args):
    '''
    CMNIST naming : xxxx_0_0.png where first 0 is the label and second 0 is the bias attribute (color)
    CCIFAR10 naming : xxxx_0_0.png where first 0 is the label (car, plane, etc) and second 0 is the bias attribute (noise)
    '''
    if args.exp == 'new-cmnist' or 'cmnist':
        args.w, args.h = 28, 28
        args.lr = 0.001 # original 0.001
        args.batch = 256
        args.num_classes = 10
    if args.exp == 'cifar10c':
        args.w, args.h = 32, 32
        args.lr = 0.001
        args.batch = 128
        args.num_classes = 10
    if args.exp == 'bffhq':
        args.w, args.h = 256, 256
        args.lr = 0.0001
        args.batch = 128
        args.num_classes = 2
    if args.exp == 'bar':
        args.w, args.h = 256, 256
        args.lr = 0.0001
        args.batch = 64
        args.num_classes = 6
        

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Training Biased Classifier")

    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--root", type=str, default="/home/seanko/aaai/data/debias/bffhq/", help="Dataset root")
    parser.add_argument("--save_root", type=str, default="/home/seanko/aaai/save/bffhq/", help="where to save model")

    parser.add_argument("--exp", type=str, default='bffhq', help="Dataset name")     # new-cmnist/bffhq/cifar10c/bar
    parser.add_argument("--pct", type=str, default="0.5pct", help="Percent name")
    parser.add_argument("--etc", type=str, default='vanilla', help="Experiment name")
    # parser.add_argument("--loss", type=str, required=True)          # GCE || CE
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--scheduler",  action='store_true', help='Using scheduler')
    parser.add_argument("--pretrained", default=False, help='Using Imagenet Pretrained')
    
    args = parser.parse_args()
    set_args(args)

    root = args.root
    args.image_size = (args.w, args.h)
    args.image_shape = (3, args.w, args.h)
    
    args.lr_decay_rate = 0.01 # 0.01
    args.lr_decay_schedule = [40, 60, 80, 100]
    
    args.data_root = f'{root}/{args.exp}/'

    args.save_root = f'{args.save_root}/{args.exp}-{args.pct}-{args.etc}-label_reversed/'

    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root)
        
    model = call_by_name(args)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = AdamP(model.parameters(), lr=args.lr)
    
    criterion_GCE = GeneralizedCELoss().cuda() # GCE Loss
    criterion_CE  = nn.CrossEntropyLoss().cuda() # CE Loss

   ### BUILD DATASET
    train_align    = [y for x in os.walk(f"/home/seanko/aaai/data/debias/{args.exp}/{args.pct}") for y in glob(os.path.join(x[0], 'align/*/*.png'))]
    train_conflict = [y for x in os.walk(f'/home/seanko/aaai/data/debias/{args.exp}/{args.pct}') for y in glob(os.path.join(x[0], 'conflict/*/*.png'))]
    
    train_data =   train_align + train_conflict
    

    if args.exp == 'cmnist':
        valid_data = [y for x in os.walk(f'{args.root}/{args.pct}') for y in glob(os.path.join(x[0], 'valid/*.png'))]
    
    elif args.exp == 'cifar10c':
        valid_data = [y for x in os.walk(f'{args.root}/{args.pct}') for y in glob(os.path.join(x[0], 'valid/*/*.png'))]

    else:
        valid_data = [y for x in os.walk(f'{args.root}') for y in glob(os.path.join(x[0], 'valid/*.png'))]

    if args.exp == 'bffhq':
        test_data  = [y for x in os.walk(f'{args.root}') for y in glob(os.path.join(x[0], 'test/*.png'))]

    else:
        test_data  = [y for x in os.walk(f'{args.root}') for y in glob(os.path.join(x[0], 'test/*/*.png'))]


    print(args.pct, 'pct')
    print(len(train_data), 'train length')
    print(len(valid_data), 'valid length')
    print(len(test_data), 'test length')
    

    '''
    bffhq 0_1 = young_men  // 0_0 = young_women
          1_1 = old_men   // 1_0 = old_women
          young/old = label (front number)  //  men/women = attribute (back number)
          
    0918
    hence, we both look at .split('_')[-2]
    '''
    
    # cmnist 0_0 = (0, red) // 0_1 = (0, blue)




    # if args.exp == 'bar' or args.exp == 'bffhq':
    #     label_attr = np.array([int(each.split('_')[-1][0]) for each in test_data])
    #     bias_attr = np.array([int(each.split('_')[-2]) for each in test_data])
        
        
    # else: # cmnist, cifar10c
    label_attr = np.array([int(each.split('_')[-2]) for each in test_data])
    bias_attr = np.array([int(each.split('_')[-1][0]) for each in test_data])
         
        # label_attr_val = np.array([int(each.split('_')[-2]) for each in valid_data])
        # bias_attr_val = np.array([int(each.split('_')[-1][0]) for each in valid_data])
        
    test_align = np.array(test_data)[label_attr == bias_attr]
    test_conflict = np.array(test_data)[label_attr != bias_attr]


    
    train_transform, valid_transform = get_transform(args)
    trainSet = BaseDataset(train_data, args, transform=train_transform, mode='train')
    validSet = BaseDataset(valid_data, args, transform=valid_transform, mode='valid')
    testSet_align = BaseDataset(test_align, args, transform=valid_transform, mode='valid')
    testSet_conflict = BaseDataset(test_conflict, args, transform=valid_transform, mode='valid')


    train_loader = torch.utils.data.DataLoader(trainSet, batch_size=args.batch, shuffle=True, drop_last=False, num_workers=args.num_workers)
    valid_loader = torch.utils.data.DataLoader(validSet, batch_size=args.batch, shuffle=False, drop_last=False, num_workers=4)
    bias_test_loader  = torch.utils.data.DataLoader(testSet_align, batch_size=args.batch, shuffle=False, drop_last=False, num_workers=4)
    unbias_test_loader  = torch.utils.data.DataLoader(testSet_conflict, batch_size=args.batch, shuffle=False, drop_last=False, num_workers=4)


    res = {'train_accuracy':[],'train_loss':[], 'valid_accuracy':[],'valid_loss':[], 'bias_test_accuracy':[],'unbias_test_accuracy':[], 'total_test_accuracy':[]}
    meter = CustomMeter(res)

    ###################################################
    ###################################################
    # Phase 1: Training Biased Classifier
    ###################################################
    ###################################################


    valid_best = 0
    for epoch in range(1, args.epoch+1):
        
        if args.scheduler:
            adjust_learning_rate(optimizer, epoch, args)
        
        # Training Process
        train_corr, train_loss = 0, 0
        model.train()
        for iter_idx, (sample_idx, inputs, labels, _) in tqdm(enumerate(train_loader), total=len(train_loader)):
            if args.exp =='cmnist':
                inputs  = inputs.flatten(1).cuda(non_blocking=True)    
            else:
                inputs  = inputs.cuda(non_blocking=True)
                
            labels  = labels.cuda(non_blocking=True)

            outputs = model(inputs)
            loss = criterion_GCE(outputs, labels).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_corr += (outputs.argmax(1) == labels).float().sum().item()
            train_loss += loss.item()

        # Valid Process
        model.eval()
        with torch.no_grad():
            valid_corr, valid_loss = 0, 0
            for iter_idx, (_, inputs, labels, _) in enumerate(valid_loader):
                if args.exp =='cmnist':
                    inputs  = inputs.flatten(1).cuda(non_blocking=True)    
                else:
                    inputs  = inputs.cuda(non_blocking=True)

                labels = labels.cuda()
                outputs = model(inputs)
                loss = criterion_GCE(outputs, labels).mean()          

                valid_corr += (outputs.argmax(1) == labels).float().sum().item()
                valid_loss += loss.item()

        train_accuracy = train_corr/len(train_loader.dataset)
        valid_accuracy = valid_corr/len(valid_loader.dataset)
        print(f'Current valid accuracy: {valid_accuracy}')
        
        if valid_accuracy > valid_best:
            print(f'\t Best valid accuracy: {valid_best} -> {valid_accuracy}, epoch: {epoch}')
            valid_best = valid_accuracy
            save('best_model', epoch, args.save_root, model, False)
        save('last', epoch, args.save_root, model, False)

        # Test Process
        with torch.no_grad():
            bias_test_corr = 0
            for iter_idx, (_, inputs, labels, _) in enumerate(bias_test_loader):
                
                if args.exp =='cmnist':
                    inputs  = inputs.flatten(1).cuda(non_blocking=True)    
                else:
                    inputs  = inputs.cuda(non_blocking=True)

    
                labels = labels.cuda()
                outputs = model(inputs)
                loss = criterion_GCE(outputs, labels).mean()             
                bias_test_corr += (outputs.argmax(1) == labels).float().sum().item()
            bias_test_accuracy = bias_test_corr/len(bias_test_loader.dataset)

            unbias_test_corr = 0
            for iter_idx, (_, inputs, labels, _) in enumerate(unbias_test_loader):
                if args.exp =='cmnist':
                    inputs  = inputs.flatten(1).cuda(non_blocking=True)    
                else:
                    inputs  = inputs.cuda(non_blocking=True)

                labels = labels.cuda()
                outputs = model(inputs)
                loss = criterion_CE(outputs, labels).mean()             
                unbias_test_corr += (outputs.argmax(1) == labels).float().sum().item()
                
            unbias_test_accuracy = unbias_test_corr/len(unbias_test_loader.dataset) 
            total_len = len(unbias_test_loader.dataset) + len(bias_test_loader.dataset)
            total_test_accuracy = (bias_test_corr+unbias_test_corr) / total_len

            # if total_test_accuracy > valid_best:
            #     print(f'\t Best valid accuracy: {valid_best} -> {valid_accuracy}, epoch: {epoch}')
            #     valid_best = total_test_accuracy
            #     save('best', epoch, args.save_root, model, False)
            # save('last', epoch, args.save_root, model, False)

        print(f'Current test accuracy: [{bias_test_accuracy}|{unbias_test_accuracy} | {total_test_accuracy}] | epoch: {epoch}')

    
        # Stats on Board
        meter.update('train_accuracy', train_accuracy)
        meter.update('train_loss', train_loss/len(train_loader))
        meter.update('valid_accuracy', valid_accuracy)
        meter.update('valid_loss', valid_loss/len(valid_loader))
        meter.update('bias_test_accuracy', bias_test_accuracy)
        meter.update('unbias_test_accuracy', unbias_test_accuracy)
        meter.update('total_test_accuracy', total_test_accuracy)

        meter.save(args.save_root, 'results')

