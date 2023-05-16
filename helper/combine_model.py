from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import torch
from torch.utils.data import DataLoader
from train.data_loader_all import MIL as MIL
from train.model_all import Model
import os
import numpy as np
import pandas as pd
from os.path import join as ospj
from argparse import ArgumentParser

def combine_model(args, order, loader, _1pNet, _19qNet, do='save', cross_val=True):
    if do == 'load':
        # load logistic model
        if cross_val == True:
            logistic_model = torch.load(ospj('model', args.dname_1pNet + '_' + args.dname_19qNet, \
                                                'logistic', str(order), 'logistic_model.pth'))
        else:
            logistic_model = torch.load(ospj('model', args.dname_1pNet + '_' + args.dname_19qNet, \
                                              'logistic', 'all', 'logistic_model.pth'))
        return logistic_model
    elif do == 'save':
        GT_CLASS = []
        NGS_1 = []
        NGS_19 = []
        PREDICT_1 = []
        PREDICT_19 = []
        for slide_order, (bag_feature, bag_label, ngs_1, ngs_19, slide) in enumerate(loader):
            
            bag_feature = bag_feature.to(torch.float32).to(args.device)   
            BS, T, F = bag_feature.shape
            
            batch_logits_1, tile_score_1 = _1pNet(bag_feature)       
            batch_logits_19, tile_score_19 = _19qNet(bag_feature)       
            
            predict_1 = batch_logits_1.squeeze(1)
            predict_19 = batch_logits_19.squeeze(1)
            
            bag_label = bag_label.to(torch.float32).to(args.device)
            
            NGS_1.extend(ngs_1.cpu().tolist())
            NGS_19.extend(ngs_19.cpu().tolist())
            GT_CLASS.extend(bag_label.cpu().tolist())
            PREDICT_1.extend(predict_1.cpu().tolist())
            PREDICT_19.extend(predict_19.cpu().tolist())
        
        train_pd = pd.DataFrame({'GT_CLASS':GT_CLASS, 'PREDICT_1':PREDICT_1, 'PREDICT_19':PREDICT_19})
        train_features = train_pd[['PREDICT_1', 'PREDICT_19']]
        train_targets = train_pd['GT_CLASS']
        
        logistic_model = LogisticRegression() 
        logistic_model.fit(train_features, train_targets)

        # save logistic model (do not use state_dict())
        if cross_val == True:
            dname_logistic = ospj('model', args.dname_1pNet + '_' + args.dname_19qNet, 'logistic', str(order))
            os.makedirs(dname_logistic, exist_ok=True)
            torch.save(logistic_model, ospj(dname_logistic, 'logistic_model.pth'))
        else:
            dname_logistic = ospj('model', args.dname_1pNet + '_' + args.dname_19qNet, 'logistic', 'all')
            os.makedirs(dname_logistic, exist_ok=True)
            torch.save(logistic_model, ospj(dname_logistic, 'logistic_model.pth'))

