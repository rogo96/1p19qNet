import os
from os.path import join as ospj
from munch import Munch
import pandas as pd
from numpy import indices
import h5py
from tqdm import tqdm, trange
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, plot_roc_curve, recall_score, precision_score, f1_score, roc_curve, auc


class Trainer():
    def __init__(self, args, dname_net, cross_val=True):
        # cross_validation 
        # parent_dir = ospj('model', )
        if cross_val == True:
            n_time = args.seed_num
            self.dname_model = ospj('model', args.dname_1pNet + '_' + args.dname_19qNet, dname_net, str(n_time))
            self.dname_log = ospj('log', args.dname_1pNet + '_' + args.dname_19qNet, dname_net, str(n_time))
        # all data (for external validation evaluation)
        elif cross_val == False:
            self.dname_model = ospj('model', args.dname_1pNet + '_' + args.dname_19qNet, dname_net, 'all')
            self.dname_log = ospj('log', args.dname_1pNet + '_' + args.dname_19qNet, dname_net, 'all')

        os.makedirs(self.dname_model, exist_ok=True)
        os.makedirs(self.dname_log, exist_ok=True)
        self.args = args
        self.writer = SummaryWriter(log_dir=self.dname_log) 
        self.criterion = nn.MSELoss()

    def train(self, model, optimizer, train_loader, val_loader):
        print('training starts!')
        worst_val_auc = -1
        for epoch in trange(1, self.args.num_epochs+1):
            samples = 0
            total_loss = 0
            total_correct = 0
            train_offset, train_log = self.train_epoch(model, optimizer, train_loader)
            offset, val_log = self.val_epoch(model, val_loader)

            if (val_log['val_auc_value'] > worst_val_auc) :
                worst_val_auc = val_log['val_auc_value']
                print('\n best AUC_model save')
                if len(os.listdir(self.dname_model)) != 0:
                    file_name = os.listdir(self.dname_model)[0]
                    os.remove(ospj(self.dname_model, file_name))
                torch.save(model.state_dict(), ospj(self.dname_model, 'best_model_' + str(epoch) + '.pth'))
            
            
            for k, v in train_log.items():
                self.writer.add_scalar(k, v, epoch)
            for k, v in val_log.items():
                self.writer.add_scalar(k, v, epoch)
            
        
    def train_epoch(self, model, optimizer, train_loader):
        args = self.args
        model.train()
        loss_epoch = 0
        PREDICT = []
        GT_CLASS = []

        log = Munch()

        for i, (bag_feature, bag_label, ngs_1, ngs_19, slide) in enumerate(tqdm(train_loader, leave=False)):
            bag_feature = bag_feature.to(torch.float32).to(args.device)   
            BS, T, F = bag_feature.shape       
            batch_logits, tile_score = model(bag_feature)   
            predict = batch_logits.squeeze(1)                    
                    
            bag_label = bag_label.to(torch.float32).to(args.device)
            ngs_1 = ngs_1.to(torch.float32).to(args.device)    
            ngs_19 = ngs_19.to(torch.float32).to(args.device)
            GT_CLASS.extend(bag_label.cpu().tolist())
            PREDICT.extend(predict.cpu().tolist())
            
            if args.ngs_1_19 == 1:
                loss = self.criterion(predict, ngs_1)
            elif args.ngs_1_19 == 19:
                loss = self.criterion(predict, ngs_19)
            print(f'bag label : {bag_label}, predict : {predict}, loss : {loss}')
            # if loss has nan value, stop training
            if torch.isnan(loss):
                raise ValueError('train loss is nan, check github data preprocessing part (excel)')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch += loss.detach()
            
        fper, tper, thresholds = roc_curve(GT_CLASS, PREDICT, pos_label=0)
        auc_value = auc(fper, tper)
        log.train_auc_value = auc_value
        log.train_loss = loss_epoch
        
        return i, log
    
    @torch.no_grad()
    def val_epoch(self, model, val_loader):
        args = self.args
        model.eval()
        loss_epoch = 0
        PREDICT = []
        GT_CLASS = []
        
        log = Munch()
            
        for i, (bag_feature, bag_label, ngs_1, ngs_19, slide) in enumerate(tqdm(val_loader, leave=False)):
            bag_feature = bag_feature.to(torch.float32).to(args.device)   
            BS, T, F = bag_feature.shape       
            batch_logits, tile_score = model(bag_feature)   
            predict = batch_logits.squeeze(1)  
                    
            bag_label = bag_label.to(torch.float32).to(args.device)
            ngs_1 = ngs_1.to(torch.float32).to(args.device)  
            ngs_19 = ngs_19.to(torch.float32).to(args.device)
            GT_CLASS.extend(bag_label.cpu().tolist())
            PREDICT.extend(predict.cpu().tolist())
            
            if args.ngs_1_19 == 1:
                loss = self.criterion(predict, ngs_1)
            elif args.ngs_1_19 == 19:
                loss = self.criterion(predict, ngs_19)

            # if loss has nan value, stop training
            if torch.isnan(loss):
                raise ValueError('valid loss is nan')
                
            loss_epoch += loss.detach()
            
        fper, tper, thresholds = roc_curve(GT_CLASS, PREDICT, pos_label=0)
        auc_value = auc(fper, tper)
        
        log.val_auc_value = auc_value
        log.val_loss = loss_epoch
        
        return i, log
        
