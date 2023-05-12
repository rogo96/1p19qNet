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
import math
import h5py
import natsort

class MIL(Dataset):
    def __init__(self, split= "test", data_dir='feature_dir', seed_num = 1):
        self.data_dir = data_dir
        self.split = split
        
        torch.manual_seed(seed_num)
        torch.cuda.manual_seed(seed_num)
        torch.cuda.manual_seed_all(seed_num)
        np.random.seed(seed_num)
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(seed_num)
        
        
        ### data split
        # random.seed(seed_num)
        
        self.dname_dataset = ospj('/home/dhlee/Chowder_Practice/feature', self.data_dir, 'total')
        
        self.malignity = os.listdir(ospj('/home/dhlee/Chowder_Practice/feature', self.data_dir, 'malignity'))
        self.malignity = natsort.natsorted(self.malignity)
        random.shuffle(self.malignity)
        train_mal_num = math.ceil(len(self.malignity) * 0.6)
        val_mal_num = math.ceil(len(self.malignity) * 0.2)
        
        all_files = os.listdir(self.dname_dataset)
        all_files = natsort.natsorted(all_files)
        
        random.shuffle(all_files)
        
        train_file_num = math.ceil(len(all_files) * 0.6)
        val_file_num = math.ceil(len(all_files) * 0.2)
        
        train_files = all_files[ : train_file_num] + self.malignity[:train_mal_num]
        val_files = all_files[train_file_num : (train_file_num + val_file_num)] + self.malignity[train_mal_num: (train_mal_num + val_mal_num)]
        test_files = all_files[(train_file_num + val_file_num) : ] + self.malignity[(train_mal_num + val_mal_num) : ]

        if self.split == 'train':
            # FIXME: x[-3]은 svs일때만 임 바꿔줘야함
            self.slides = [x[:-3] for x in train_files]
            
        elif self.split == 'val':
            self.slides = [x[:-3] for x in val_files]
            
        elif self.split == 'test':
            self.slides = [x[:-3] for x in test_files]

        elif self.split == 'all':
            self.slides = [x[:-3] for x in all_files]
        
        df = pd.read_excel("/home/dhlee/Chowder_Practice/excel/1230LDH.xlsx", engine="openpyxl")
        self.malignity = df.loc[df['Grade'] == 4]["Serial Number"].tolist()
        self.normal = [x for x in df["Serial Number"].tolist() if x not in self.malignity]

        self.df = df.loc[:,["Serial Number", "NGS1", "NGS19", "Class"]] 
        self.slide2label = {}
        for index, row in self.df.iterrows():
            slide = row['Serial Number']
            label = row['Class']
            ngs_1 = row['NGS1']
            ngs_19 = row['NGS19']
            self.slide2label[slide] = [label, ngs_1, ngs_19]
            
    def __len__(self):
        return len(self.slides)
    
    def __getitem__(self, index):
        slide = self.slides[index]
        
        label = self.slide2label[slide][0]
        ngs_1 = self.slide2label[slide][1]
        ngs_19 = self.slide2label[slide][2]
        
        h5_file = slide + '.h5'
        if slide in self.normal:
            h5_file = ospj(self.dname_dataset, h5_file)
        elif slide in self.malignity:
            h5_file = ospj('/home/dhlee/Chowder_Practice/feature', self.data_dir, 'malignity', h5_file)
        with h5py.File(h5_file, "r") as hdf:
            hdf_np = np.array(hdf['df']['block0_values'], dtype=np.float32)

        return hdf_np, label, ngs_1, ngs_19, slide
