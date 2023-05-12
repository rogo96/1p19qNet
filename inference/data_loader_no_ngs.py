import os
from os.path import join as ospj
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import torch.backends.cudnn as cudnn
import random
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
import torch
import numpy as np
import pickle
from tqdm import tqdm
import math
import h5py
import albumentations as A
import natsort

class External_validation(Dataset):
    def __init__(self, data_dir='feature_dir'):
        self.data_dir = data_dir
        
        # torch.manual_seed(seed_num)
        # torch.cuda.manual_seed(seed_num)
        # torch.cuda.manual_seed_all(seed_num)
        # np.random.seed(seed_num)
        # cudnn.benchmark = False
        # cudnn.deterministic = True
        # random.seed(seed_num)
        
        
        ### data split
        # random.seed(seed_num)
        
        
        self.dname_dataset = ospj('/home/dhlee/Chowder_Practice/feature', self.data_dir, 'total')
        all_files = os.listdir(self.dname_dataset)
        all_files = natsort.natsorted(all_files)
        # random.shuffle(all_files)
        
        # train_file_num = math.ceil(len(all_files) * 0.6)
        # val_file_num = math.ceil(len(all_files) * 0.2)
        
        # # train_files = all_files[ : train_file_num] 
        # # val_files = all_files[train_file_num : (train_file_num + val_file_num)] 
        # # test_files = all_files[(train_file_num + val_file_num) : ] 
        
        # train_files = all_files
        # val_files = all_files
        # test_files = all_files
        self.slides = [x[:-3] for x in all_files]
        # print(f'file num : {len(test_files)}')
        # if self.split == 'train':
        #     self.slides = [x[:-3] for x in train_files]
        # elif self.split == 'val':
        #     self.slides = [x[:-3] for x in val_files]
        # elif self.split == 'test':
        
        
        # TODO: make excel file (data-preprocessing)        
        df = pd.read_excel("/home/dhlee/Chowder_Practice/excel/TCGA_remove_6.xlsx", engine="openpyxl")
        self.df = df.loc[:,["Serial Number", "Class"]] 
        self.slide2label = {}
        for index, row in self.df.iterrows():
            slide = row['Serial Number']
            label = row['Class']
            self.slide2label[slide] = [label]
            
    def __len__(self):
        return len(self.slides)
    
    def __getitem__(self, index):
        slide = self.slides[index]
        label = self.slide2label[slide][0]

        # TODO: make h5 file (data-preprocessing)
        h5_file = ospj(self.dname_dataset, slide + '.h5')
        hdf = h5py.File(h5_file, "r")
        hdf_np = np.array(hdf['df']['block0_values'], dtype=np.float32)
        hdf.close()
        return hdf_np, label, slide
        
# if __name__ == "__main__" :
    
#     loader = DataLoader(MIL(split="test",data_dir='tcga_feature', seed_num = 1)
#                         , batch_size=1, shuffle=False)
    
#     print(len(loader))
#     for i, (hdf_tensor, label, slide) in enumerate(loader):
#         print(hdf_tensor.shape, label, slide)
#         hdf_tensor = hdf_tensor.to(torch.float32)   # (BS x N_tiles x 2048)
#         B, T, F = hdf_tensor.shape 