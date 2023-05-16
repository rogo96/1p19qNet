import os
from os.path import join as ospj
from torch.utils.data import Dataset
import pandas as pd
import torch.backends.cudnn as cudnn
import random
import torch
import numpy as np
import math
import h5py
import natsort
import glob

class MIL(Dataset):
    def __init__(self, split= "test", feat_dir='feature_dir', seed_num = 1, inf=False):
        self.feat_dir = feat_dir
        self.split = split
        self.inf = inf
        
        torch.manual_seed(seed_num)
        torch.cuda.manual_seed(seed_num)
        torch.cuda.manual_seed_all(seed_num)
        np.random.seed(seed_num)
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(seed_num)
        
        self.dname_dataset = ospj(self.feat_dir, 'total')
        all_files = os.listdir(self.dname_dataset)
        all_files = natsort.natsorted(all_files)
        
        random.shuffle(all_files)
        train_file_num = math.ceil(len(all_files) * 0.3)
        val_file_num = math.ceil(len(all_files) * 0.3)
        
        train_files = all_files[ : train_file_num]
        val_files = all_files[train_file_num : (train_file_num + val_file_num)]
        test_files = all_files[(train_file_num + val_file_num) : ]

        if self.split == 'train':
            self.slides = [x.split('.')[0] for x in train_files]
        elif self.split == 'val':
            self.slides = [x.split('.')[0] for x in val_files]
        elif self.split == 'test':
            self.slides = [x.split('.')[0] for x in test_files]
        elif self.split == 'all':
            self.slides = [x.split('.')[0] for x in all_files]
        
        excel_path = glob.glob(ospj(self.feat_dir, '*.xlsx'))[0]
        df = pd.read_excel(excel_path, engine="openpyxl")

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
        
        h5_file = slide + '.h5'
        h5_file = ospj(self.dname_dataset, h5_file)
        with h5py.File(h5_file, "r") as hdf:
            hdf_np = np.array(hdf['df']['block0_values'], dtype=np.float32)

        if self.inf == False:
            ngs_1 = self.slide2label[slide][1]
            ngs_19 = self.slide2label[slide][2]
            return hdf_np, label, ngs_1, ngs_19, slide
        else:
            return hdf_np, label, slide
