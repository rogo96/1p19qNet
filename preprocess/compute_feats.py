import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, glob, copy
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
        feats = self.feature_extractor(x) # Bs x 2048
        c = self.fc(feats.view(feats.shape[0], -1)) # BS x 2
        return feats.view(feats.shape[0], -1), c    # BS x 2048,  BS x 2

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
                                        # stain_normalize(normalizer),
                                        ToTensor(),
                                        # RGBshift(),
                                    ]))
    dataloader = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    return dataloader, len(transformed_dataset)

def compute_feats(args, bags_list, i_classifier, total_list, save_path=None):
    i_classifier.eval()
    num_bags = len(bags_list)

    for i in tqdm(range(0, num_bags)):
        # if bags_list[i].split(os.path.sep)[-1] in temp:
        #     continue
        feats_list = []
        binary_file_path = glob.glob(os.path.join(bags_list[i], '*.jpeg'))
        # binary_file_path = glob.glob(os.path.join(bags_list[i], '*.jpeg'))
        
        # dataloader, bag_size =  bag_dataset(args, binary_file_path, normalizer)
        dataloader, bag_size =  bag_dataset(args, binary_file_path)
        
        col_row = []
        with torch.no_grad():
            slide_name = bags_list[i].split(os.path.sep)[-1]
            print('\n',slide_name + '.h5')
            for iteration, (batch, tile_name) in enumerate(dataloader):
                # if i == 0 and iteration == 0:
                #     ex = make_grid(batch['input'], nrow=8)
                #     # save_image(ex, args.output_dataset + '.jpg')
                #     save_image(ex, 'tcga_stain_norm_rgb_norm' + '.jpg')
                #     break
                for idx in range(len(tile_name)):
                    col = int(tile_name[idx].split('/')[-1].split('_')[1])
                    row = int(tile_name[idx].split('/')[-1].split('_')[2].split('.')[0])
                    col_row.append([col, row])
                
                patches = batch['input'].float().cuda() 
                feats, classes = i_classifier(patches)
                feats = feats.cpu().numpy()
                feats_list.extend(feats)
                sys.stdout.write('\r Computed: {}/{} -- {}/{}'.format(i+1, num_bags, iteration+1, len(dataloader)))
        ### print(feats_list)
        # print(f'col_row: {col_row[0]}')    
        
        with open(os.path.join('/data1/wsi/new_data/pickle', args.pickle, slide_name + '.pickle'),'wb') as fw:
            pickle.dump(col_row, fw)       
        # dh 0122
        if len(feats_list) == 0:
            print('No valid patch extracted from: ' + bags_list[i])
        else:
            #    밑꺼살려
            df = pd.DataFrame(feats_list)
            
            os.makedirs(os.path.join(save_path, 'total'), exist_ok=True)

            ### os.makedirs(os.path.join(save_path, 'all_train'), exist_ok=True)
            ### os.makedirs(os.path.join(save_path, 'all_val'), exist_ok=True)
            ### os.makedirs(os.path.join(save_path, 'all_test'), exist_ok=True)
            
            #     밑꺼살려
            for file in total_list:
                if bags_list[i].split(os.path.sep)[-1] in file:
                    print('\n', save_path, bags_list[i].split(os.path.sep)[-1])
                    df.to_hdf(os.path.join(save_path, 'total', bags_list[i].split(os.path.sep)[-1]+'.h5'), key='df', mode='w')
                    break
                
                
                
          

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size of dataloader [128]')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of threads for datalodaer')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--input_dataset', default='prepared_dataset', type=str)
    parser.add_argument('--output_dataset', default='feature', type=str)
    parser.add_argument('--pickle', default='1225_pickle', type=str)
    args = parser.parse_args()
    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(x) for x in gpu_ids)
    
    resnet = models.resnet50(pretrained=True)
    num_feats = 2048
    
    for param in resnet.parameters():
        param.requires_grad = False
    resnet.fc = nn.Identity()
    
    i_classifier = IClassifier(resnet, num_feats, output_class=args.num_classes).cuda()
    
    # bags_path = os.path.join('/home/dhlee/Chowder_Practice/WSI', args.dataset, 'single', '*', '*')
    # bags_path = os.path.join('/data1/wsi/temp', '*', '*')
    
    # bags_path = os.path.join('/data1/wsi', args.input_dataset, '*', '*')
    
    # bags_path = os.path.join('/data1/TCGA_LGG/tiles', '*', '*')
    # bags_path = os.path.join('/home/dhlee/Chowder_Practice/test_tiles', '*')
    bags_path = os.path.join('/data1/wsi/new_data/tile/wsi', '*')
    # bags_path = os.path.join('/data1/wsi/new_data/tile/0127_norm_tcga', '*')
    
    
    
    
    
    # bags_path = os.path.join('/data1/TCGA_LGG/1209_feature/TCGA_LGG', '*')
    
    
    
    
    bags_list = glob.glob(bags_path)
    # feats_path = os.path.join('/data1/wsi/feature', args.output_dataset)
    # feats_path = os.path.join('/data1/wsi/feature/1212_feature')
    # feats_path = os.path.join('data1/TCGA_LGG/feature/1214_feature')
    feats_path = os.path.join('/data1/wsi/new_data/feature')
    
    
    os.makedirs(feats_path, exist_ok=True)
    
    # total_list = os.listdir('/data1/w[s]i/prepared_dataset/total')
    # total_list = os.listdir('/data1/wsi/abnormal_tile/wsi')
    # total_list = os.listdir('/data1/TCGA_LGG/total/Astrocytoma') + os.listdir('/data1/TCGA_LGG/total/Oligodendroglioma')
    # total_list = os.listdir('/home/dhlee/Chowder_Practice/test_tiles')
    total_list = os.listdir('/data1/wsi/new_data/tile/wsi')
    # total_list = os.listdir('/data1/wsi/new_data/tile/0127_norm_tcga')
    
    
    # total_list = os.listdir('/data1/wsi/temp/wsi') 
        
    # total_list = os.listdir('/data1/wsi/temp')
    
    
    # train_list = os.listdir('/data1/wsi/prepared_dataset/all_train')
    # val_list = os.listdir('/data1/wsi/prepared_dataset/all_val')
    # test_list = os.listdir('/data1/wsi/prepared_dataset/all_test')
    # compute_feats(args, bags_list, i_classifier, train_list, val_list, test_list, feats_path)
    
    
    # os.makedirs('/data1/wsi/feature/1109_pickle',exist_ok=True)
    # os.makedirs('/data1/wsi/feature/col_row_pick_1212',exist_ok=True)
    os.makedirs(os.path.join('/data1/wsi/new_data/pickle', args.pickle),exist_ok=True)
    
    
    compute_feats(args, bags_list, i_classifier, total_list, feats_path)
    
    # n_classes = glob.glob(os.path.join('datasets', args.dataset, '*'+os.path.sep))
    # n_classes = sorted(n_classes)
    # all_df = []
    # for i, item in enumerate(n_classes):
    #     bag_h5 = glob.glob(os.path.join(item, '*.h5'))
    #     bag_hdf = pd.read_hdf(bag_h5,'s')
    #     # bag_hdf['label'] = i
    #     bag_hdf.to_hdf(os.path.join('datasets', args.dataset, item.split(os.path.sep)[2]+'.h5'), key='df', mode='w')
    # bags_path = pd.concat(all_df, axis=0, ignore_index=True)
    # bags_path = shuffle(bags_path)
    # bags_path.to_hdf(os.path.join('datasets', args.dataset, key='df', mode='w'))
