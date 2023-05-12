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
from memory_profiler import profile 


class MIL_ours(Dataset):
    def __init__(self, split= "test", n_time=0, data_dir='feature_dir', seed_num = 1, infer=False, transform=None):
        self.data_dir = data_dir
        self.split = split
        self.infer = infer
        self.transform = transform
        
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
        
        # # self.dname_dataset = ospj(ospj('/home/dhlee/Chowder_Practice/feature', 'only_tensor', 'total'))
        all_files = os.listdir(self.dname_dataset)
        all_files = natsort.natsorted(all_files)
        # # print(all_files)
        
        random.shuffle(all_files)
        
        train_file_num = math.ceil(len(all_files) * 0.6)
        val_file_num = math.ceil(len(all_files) * 0.2)
        
        # train_files = all_files[ : train_file_num] 
        # val_files = all_files[train_file_num : (train_file_num + val_file_num)] 
        # test_files = all_files[(train_file_num + val_file_num) : ] 
        
        train_files = all_files[ : train_file_num] + self.malignity[:train_mal_num]
        val_files = all_files[train_file_num : (train_file_num + val_file_num)] + self.malignity[train_mal_num: (train_mal_num + val_mal_num)]
        test_files = all_files[(train_file_num + val_file_num) : ] + self.malignity[(train_mal_num + val_mal_num) : ]
        
        # for all model
        all = all_files + self.malignity

        # train_files = all_files
        # val_files = all_files
        # test_files = all_files + self.malignity


        
        # # self.split = 'all_' + split
        if self.split == 'train':
            # self.dname_dataset= ospj('/home/dhlee/Chowder_Practice/feature', self.data_dir, 'train')
            # train_files = os.listdir(self.dname_dataset)
            self.slides = [x[:-3] for x in train_files]
            # print(self.slides)
            # print('loader train files ', train_files)
            
        elif self.split == 'val':
            # self.dname_dataset= ospj('/home/dhlee/Chowder_Practice/feature', self.data_dir, 'val')
            # val_files = os.listdir(self.dname_dataset)
            self.slides = [x[:-3] for x in val_files]
            # print('loader val ',val_files)
            
        elif self.split == 'test':
            # self.dname_dataset= ospj('/home/dhlee/Chowder_Practice/feature', self.data_dir, 'test')
            # test_files = os.listdir(self.dname_dataset)
            self.slides = [x[:-3] for x in test_files]
            # print(f'test : {len(test_files)}')
            # print(f'test : {self.slides}')
            # print('loader test ', test_files)
        elif self.split == 'all':
            self.slides = [x[:-3] for x in all]
            print('using all data')
        # read excel
        # if 'prior' in self.data_dir:
            # df = pd.read_excel("0314sUYJ.xlsx", engine="openpyxl")
        # if 'after' in self.data_dir:
        # else:
        # df = pd.read_excel("1020LDH.xlsx", engine="openpyxl")
        # df = pd.read_excel("0725UYJ.xlsx", engine="openpyxl")
        # df = pd.read_excel("0725_revise.xlsx", engine="openpyxl")
        # df = pd.read_excel("1020LDH.xlsx", engine="openpyxl")
        # 238개 1109
        # df = pd.read_excel("1109_revise.xlsx", engine="openpyxl")
        # 245개 1225
        # df = pd.read_excel("1222_LDH_nomar.xlsx", engine="openpyxl")
        # df = pd.read_excel("1222LDH.xlsx", engine="openpyxl")

        
        # df = pd.read_excel("1130(malgin)LDH.xlsx", engine="openpyxl")
        
        df = pd.read_excel("/home/dhlee/Chowder_Practice/excel/1230LDH.xlsx", engine="openpyxl")
        self.malignity = df.loc[df['Grade'] == 4]["Serial Number"].tolist()
        # self.malignity = malignity["Serial Number"].tolist()
        self.normal = [x for x in df["Serial Number"].tolist() if x not in self.malignity]

        # elif 'prior' in self.data_dir:
            
        # df = pd.read_excel("0314sUYJ.xlsx", engine="openpyxl")
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
        
        # label = torch.tensor(self.slide2label[slide][0])
        # ngs_1 = torch.tensor(self.slide2label[slide][1])
        # ngs_19 = torch.tensor(self.slide2label[slide][2])
        h5_file = slide + '.h5'
        # # hdf = h5py.File(ospj(self.dname_dataset, h5_file), "r")
        if slide in self.normal:
            h5_file = ospj(self.dname_dataset, h5_file)
            # hdf = h5py.File(h5_file, "r")
        elif slide in self.malignity:
            h5_file = ospj('/home/dhlee/Chowder_Practice/feature', self.data_dir, 'malignity', h5_file)
            # hdf = h5py.File(ospj(self.dname_dataset, h5_file), "r")
        with h5py.File(h5_file, "r") as hdf:
            hdf_np = np.array(hdf['df']['block0_values'], dtype=np.float32)


        # hdf = h5py.File(h5_file, "r")

        # hdf.close()
        # hdf_np delete


        return hdf_np, label, ngs_1, ngs_19, slide
