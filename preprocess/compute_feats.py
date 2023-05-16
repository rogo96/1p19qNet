import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms.functional as VF
from torchvision import transforms
import sys, argparse, os, glob, copy, shutil
import pandas as pd
import numpy as np
from PIL import Image
from collections import OrderedDict
from sklearn.utils import shuffle
import tables
import pickle
import albumentations as A

from torchvision.utils import save_image
from torchvision.utils import make_grid
from tqdm import tqdm
import staintools

class IClassifier(nn.Module):
    def __init__(self, feature_extractor, feature_size, output_class):
        super(IClassifier, self).__init__()
        
        self.feature_extractor = feature_extractor      
        self.fc = nn.Linear(feature_size, output_class)
        
        
    def forward(self, x):
        device = x.device
        feats = self.feature_extractor(x) 
        c = self.fc(feats.view(feats.shape[0], -1)) 
        return feats.view(feats.shape[0], -1), c    

class BagDataset():
    def __init__(self, h5_file, transform=None):
    
        self.files_list = h5_file
        self.transform = transform
    def __len__(self):
        return len(self.files_list)
    def __getitem__(self, idx):
        temp_path = self.files_list[idx]
        img = os.path.join(temp_path)
        img = Image.open(img)
        sample = {'input': img}
        
        if self.transform:
            sample = self.transform(sample)
        return sample , temp_path

class stain_normalize(object):
    def __init__(self, normalizer):
        self.normalizer = normalizer
    def __call__(self, sample):
        img = sample['input']

        return {'input': img} 
        
class ToTensor(object):
    def __call__(self, sample):

        img = sample['input']
        img = VF.to_tensor(img)
        img = transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
            )(img)
        return {'input': img} 
        
class RGBshift(object):
    def __call__(self, sample):
        img = sample['input']
        R,G,B = img.split(1, dim=0)
        R = torch.sub(R, torch.min(R))
        G = torch.sub(G, torch.min(G))
        B = torch.sub(B, torch.min(B))
        
        R_gain = 1 / torch.max(R)
        G_gain = 1 / torch.max(G)
        B_gain = 1 / torch.max(B)
        
        R = torch.mul(R, R_gain)
        G = torch.mul(G, G_gain)
        B = torch.mul(B, B_gain)
        
        color_list = [R,G,B]
        img = torch.cat(color_list, dim=0)
        
        return {'input': img} 
         
        
        
        
        
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

def bag_dataset(args, h5_file_path):
    transformed_dataset = BagDataset(h5_file=h5_file_path,
                                    transform=Compose([
                                        ToTensor(),
                                    ]))
    dataloader = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    return dataloader, len(transformed_dataset)

def compute_feats(args, bags_list, i_classifier, total_list, feats_path=None, tile_poisition_path=None):
    i_classifier.eval()
    num_bags = len(bags_list)

    for i in tqdm(range(0, num_bags)):
        feats_list = []
        binary_file_path = glob.glob(os.path.join(bags_list[i], '*.jpeg'))
        dataloader, bag_size =  bag_dataset(args, binary_file_path)
        
        col_row = []
        with torch.no_grad():
            slide_name = bags_list[i].split(os.path.sep)[-1]
            print('\n',slide_name + '.h5')
            for iteration, (batch, tile_name) in enumerate(dataloader):
                for idx in range(len(tile_name)):
                    col = int(tile_name[idx].split('/')[-1].split('_')[-2])
                    row = int(tile_name[idx].split('/')[-1].split('_')[-1].split('.')[0])
                    col_row.append([col, row])
                
                patches = batch['input'].float().cuda() 
                feats, classes = i_classifier(patches)
                feats = feats.cpu().numpy()
                feats_list.extend(feats)
                sys.stdout.write('\r Computed: {}/{} -- {}/{}'.format(i+1, num_bags, iteration+1, len(dataloader)))
        
        with open(os.path.join(tile_poisition_path, slide_name + '.pickle'),'wb') as fw:
            pickle.dump(col_row, fw)    

        if len(feats_list) == 0:
            print('No valid patch extracted from: ' + bags_list[i])
        else:
            df = pd.DataFrame(feats_list)
            os.makedirs(os.path.join(feats_path, 'total'), exist_ok=True)

            for file in total_list:
                if bags_list[i].split(os.path.sep)[-1] in file:
                    df.to_hdf(os.path.join(feats_path, 'total', bags_list[i].split(os.path.sep)[-1]+'.h5'), key='df', mode='w')
                    break
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size of dataloader [128]')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of threads for datalodaer')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--tile_path', default='../Data/Tile', type=str)
    parser.add_argument('--feat_path', default='../Data/Feature', type=str)
    parser.add_argument

    args = parser.parse_args()
    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(x) for x in gpu_ids)
    
    resnet = models.resnet50(pretrained=True)
    num_feats = 2048
    
    for param in resnet.parameters():
        param.requires_grad = False
    resnet.fc = nn.Identity()
    
    i_classifier = IClassifier(resnet, num_feats, output_class=args.num_classes).cuda()
    
    # transform tile to feature
    bags_path = os.path.join(args.tile_path, '*')
    bags_list = glob.glob(bags_path)
    feats_path = args.feat_path
    tile_poisition_path = os.path.join(args.feat_path, 'Tile_position')
    os.makedirs(feats_path, exist_ok=True)
    os.makedirs(tile_poisition_path, exist_ok=True)
    total_list = os.listdir(args.tile_path)
    compute_feats(args, bags_list, i_classifier, total_list, feats_path, tile_poisition_path)

    # move excel file in your feature directory 
    excel_path = os.path.join(feats_path, '..')
    excel_file = glob.glob(os.path.join(excel_path, '*.xlsx'))
    if len(excel_file) != 1:
        raise ValueError('Need one excel file in your Data directory, check Feature directory')
    else:
        shutil.move(os.path.join(excel_file[0]), feats_path)