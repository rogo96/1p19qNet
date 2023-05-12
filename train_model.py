import os
from os.path import join as ospj
import torch
from torch.optim import Adam
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from train.data_loader_all import MIL as MIL
from train.model_all import Model
from train.trainer_all_AUC import Trainer
from helper.load_model import load_best_model
from helper.combine_model import combine_model
# from test_all import test
from argparse import ArgumentParser
import time

def train_cross_validation(args):
    train_loader = DataLoader(MIL(split="train", data_dir=args.data_dir, seed_num=args.seed_num),
                                batch_size=1,
                                num_workers=2,
                                shuffle=True)
    val_loader = DataLoader(MIL(split="val", data_dir=args.data_dir, seed_num=args.seed_num),
                                batch_size=1,
                                num_workers=2,  
                                shuffle=False)
    
    args.ngs_1_19 = 1
    model = Model(args).to(args.device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    trainer = Trainer(args, args.dname_1pNet)
    trainer.train(model, optimizer, train_loader, val_loader)

    args.ngs_1_19 = 19
    model = Model(args).to(args.device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    trainer = Trainer(args, args.dname_19qNet)
    trainer.train(model, optimizer, train_loader, val_loader)

    model_1, model_19 = load_best_model(args, order=args.seed_num)
    combine_model(args, args.seed_num, train_loader, model_1, model_19, 'save')

def train_all_data(args):
    all_loader = DataLoader(MIL(split="all", data_dir=args.data_dir),
                            batch_size=1,
                            num_workers=2,
                            shuffle=False)
    # Train 1pNet
    args.ngs_1_19 = 1
    model = Model(args).to(args.device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    trainer = Trainer(args, args.dname_1pNet, cross_val=False)
    trainer.train(model, optimizer, all_loader, all_loader)

    # Train 19qNet
    args.ngs_1_19 = 19
    model = Model(args).to(args.device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    trainer = Trainer(args, args.dname_19qNet , cross_val=False)
    trainer.train(model, optimizer, all_loader, all_loader)

    # Train combine model (1pNet + 19qNet, Logistic Regression) 
    model_1, model_19 = load_best_model(args, order=None)
    combine_model(args, None, all_loader, model_1, model_19, 'save', cross_val=False)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--dname_1pNet', type=str, default='1pNet')
    parser.add_argument('--dname_19qNet', type=str, default='19qNet')
    parser.add_argument('--max_r', type=int, default=20)
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--gpu', type=int, default=0)
    # FIXME: ngs_1_19 이름바꾸기?
    # parser.add_argument('--ngs_1_19', type=int, default=1)
    parser.add_argument('--data_dir', type=str, default='0214_feature')
    parser.add_argument('--all_data', action='store_true')
    parser.add_argument('--n_fold', type=int, default=10)
    # FIXME: data 필요함?
    parser.add_argument('--data', type=str, default='test')
    parser.add_argument('--excel', action='store_true')
    args = parser.parse_args()
    args.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    if args.all_data == False:
        for i in range(1, args.n_fold+1):
            args.seed_num = i
            train_cross_validation(args)
    else:
        train_all_data(args)

    
