"""
2023-09-01.


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

def get_transform(args):
    
    train_transform = T.Compose([
            T.Resize(args.image_size),
            T.ToTensor()
    ])

    valid_transform = T.Compose([
    T.Resize(args.image_size),
    T.ToTensor()
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

        if self.mode == 'valid-bffhq':
            label = int(origin_path.split('.')[0].split('_')[-2])
            
        # for cmnist and cifar10c
        elif self.mode == 'valid':
            label = int(origin_path.split('_')[-2])
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
    if args.exp == 'new-cmnist' or 'cmnist':
        args.w, args.h = 28, 28
        args.lr = 0.001
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
    parser.add_argument("--root", type=str, default="/home/seanko/aaai/data/debias/cifar10c", help="Dataset root")
    parser.add_argument("--save_root", type=str, default="/home/seanko/aaai/save/cifar10c", help="where the model weight was saved")

    parser.add_argument("--exp", type=str, default='cifar10c', help="Dataset name")     # new-cmnist/bffhq/bar/bar
    parser.add_argument("--data_type", type=str, default='cifar10c', help='kind of data used')
    parser.add_argument("--pct", type=str, default="5pct", help="Percent name")
    parser.add_argument("--etc", type=str, default='vanilla', help="Experiment name")
    # parser.add_argument("--loss", type=str, required=True)          # GCE || CE
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--scheduler",  action='store_true', help='Using scheduler')
    parser.add_argument("--pretrained", action='store_true', help='Using Imagenet Pretrained')
    parser.add_argument("--conflict_target", type=str, default="/home/seanko/aaai/data/top_loss_samples/")
    args = parser.parse_args()
    set_args(args)

    root = args.root
    args.image_size = (args.w, args.h)
    args.image_shape = (3, args.w, args.h)
    
    args.lr_decay_rate = 0.1
    args.lr_decay_schedule = [40, 60, 80]
    
    args.data_root = f'{root}/{args.exp}/'

    args.save_root = f'{args.save_root}/cikm/{args.exp}-{args.pct}-{args.etc}'

    model = call_by_name(args)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    criterion_GCE = GeneralizedCELoss().cuda() # GCE Loss
    criterion_CE  = nn.CrossEntropyLoss().cuda() # GCE Loss

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

    
    # print(args.pct)
    print(len(train_data))
    print(len(valid_data), 'valid length')
    # label_attr = np.array([int(each.split('_')[-2]) for each in test_data])   # [0/3, 0, 0.png] -> we choose the 2nd one which is the label
    # bias_attr = np.array([int(each.split('_')[-1][0]) for each in test_data]) # 0.png -> we choose 0 which is the bias att.


    bias_attr = np.array([int(each.split('_')[-2]) for each in test_data])   # [0/3, 0, 0.png] -> we choose the 2nd one which is the label
    label_attr = np.array([int(each.split('_')[-1][0]) for each in test_data]) # 0.png -> we choose 0 which is the bias att.

    test_align = np.array(test_data)[label_attr == bias_attr]
    test_conflict = np.array(test_data)[label_attr != bias_attr]

    train_transform, valid_transform = get_transform(args)
    trainSet = BaseDataset(train_data, args, transform=train_transform)
    validSet = BaseDataset(valid_data, args, transform=valid_transform)
    testSet_align = BaseDataset(test_align, args, transform=valid_transform)
    testSet_conflict = BaseDataset(test_conflict, args, transform=valid_transform)

    train_loader = torch.utils.data.DataLoader(trainSet, batch_size=args.batch, shuffle=True, drop_last=False, num_workers=args.num_workers)
    valid_loader = torch.utils.data.DataLoader(validSet, batch_size=args.batch, shuffle=False, drop_last=False, num_workers=2)
    bias_test_loader  = torch.utils.data.DataLoader(testSet_align, batch_size=args.batch, shuffle=False, drop_last=False, num_workers=2)
    unbias_test_loader  = torch.utils.data.DataLoader(testSet_conflict, batch_size=args.batch, shuffle=False, drop_last=False, num_workers=2)

    res = {'train_accuracy':[],'train_loss':[], 'valid_accuracy':[],'valid_loss':[], 'bias_test_accuracy':[],'unbias_test_accuracy':[]}
    meter = CustomMeter(res)

    ###################################################
    ###################################################
    # Phase 2 : Extract bias samples 
    # (samples with highest loss)
    ###################################################
    ###################################################


    valid_best = 0


    model.load_state_dict(torch.load(f'{args.save_root}/best_model.path.tar')['model_state_dict'])
    print(f'Best Model Loaded from {args.save_root}/best_model.path.tar')
    model.eval()
    
    loss_dict = {each:{
        'instance':[], 'path':[], 'loss':[], } for each in range(args.num_classes)}

    with torch.no_grad():
        for sample_idx, input, label, path  in tqdm(trainSet, total=len(trainSet)):
            model.zero_grad()
            input = input.unsqueeze(0).cuda()
            label = torch.tensor(label).type(torch.LongTensor).unsqueeze(0).cuda()
            pred  = model(input)
            loss = criterion_CE(pred,label) 
            # when we extract bias, we should use CE loss 
            # since GCE is only effective when we train the model since we backpropagate the loss during training

            label_key = label.item()

            loss_dict[label_key]['instance'].append(input.cpu())
            loss_dict[label_key]['path'].append(path)
            loss_dict[label_key]['loss'].append(loss.item())
    
    # Extracting Top K samples based on class-wise criterion
    
    if args.exp == "bffhq":
        K = 200
        
    if args.pct == "5pct":
        K = 200
        
    else:
        K = 100
        
  # Extracting Top K samples based on class-wise criterion
    conflict_target_path = f'{args.conflict_target}/{args.data_type}_cikm/{args.pct}/extracted-bias-conflict'
    target_path = f'{args.conflict_target}/{args.data_type}_cikm/{args.pct}/extracted-bias-align'

    for each_label in loss_dict.keys():
        path_list     = np.array(loss_dict[each_label]['path'])
        loss_list     = np.array(loss_dict[each_label]['loss'])

        sorted_indexs = np.argsort(loss_list)[-K:]
        sorted_align_indexs = np.argsort(loss_list)[:K]

        path_sorted = path_list[sorted_indexs].tolist()
        path_sorted_align = path_list[sorted_align_indexs].tolist()
        print('path_sorted:', path_sorted)
        print('path_sorted_len:', len(path_sorted))
        print('path_sorted_align:', path_sorted_align)
        print('path_sorted_align len:', len(path_sorted_align))
            ###########################################################################################
        print('each_label:', each_label) # each_label : 0,1,2,3,4,5,6,7,8,9
            
        os.makedirs(f'{target_path}/{each_label}/', exist_ok=True)
        os.makedirs(f'{conflict_target_path}/{each_label}/', exist_ok=True)

        
            
        for each_path in path_sorted_align:
            shutil.copy(each_path, f'{target_path}/{each_label}/')

        for each_path in path_sorted:
            shutil.copy(each_path, f'{conflict_target_path}/{each_label}/')
        