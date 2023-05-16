from os.path import join as ospj
import torch
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader
from train.data_loader_all import MIL as MIL
from train.model_all import Model
from train.trainer_all_AUC import Trainer
from helper.load_model import load_best_model
from helper.combine_model import combine_model
from test_model import inference_cross_validation, inference_bootstrap, inference
from helper.make_result import plot_confusion_matrix
from argparse import ArgumentParser

def train_cross_validation(args):
    train_loader = DataLoader(MIL(split="train", feat_dir=args.feat_dir, seed_num=args.seed_num),
                                batch_size=1,
                                num_workers=4,
                                shuffle=False)
    val_loader = DataLoader(MIL(split="val", feat_dir=args.feat_dir, seed_num=args.seed_num),
                                batch_size=1,
                                num_workers=4,  
                                shuffle=False)
    print(f'train 1pNet(cross_validation, {args.seed_num} time)')
    args.ngs_1_19 = 1
    model = Model(args).to(args.device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    trainer = Trainer(args, args.dname_1pNet, cross_val=True)
    trainer.train(model, optimizer, train_loader, val_loader)

    print(f'train 19qNet(cross_validation, {args.seed_num} time)')
    args.ngs_1_19 = 19
    model = Model(args).to(args.device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    trainer = Trainer(args, args.dname_19qNet, cross_val=True)
    trainer.train(model, optimizer, train_loader, val_loader)

    model_1, model_19 = load_best_model(args, order=args.seed_num)
    combine_model(args, args.seed_num, train_loader, model_1, model_19, 'save', cross_val=True)

def train_all_data(args):
    all_loader = DataLoader(MIL(split="all", feat_dir=args.feat_dir),
                            batch_size=1,
                            num_workers=4,
                            shuffle=False)
    # Train 1pNet
    print(f'train 1pNet(all_data)')
    args.ngs_1_19 = 1
    model = Model(args).to(args.device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    trainer = Trainer(args, args.dname_1pNet, cross_val=False)
    trainer.train(model, optimizer, all_loader, all_loader)

    # Train 19qNet
    print(f'train 19qNet(all_data)')
    args.ngs_1_19 = 19
    model = Model(args).to(args.device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    trainer = Trainer(args, args.dname_19qNet, cross_val=False)
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
    parser.add_argument('--max_r', type=int, default=100)
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--feat_dir', type=str, default='Data/Feature')
    parser.add_argument('--all_data', action='store_true')
    parser.add_argument('--n_fold', type=int, default=10)
    parser.add_argument('--boot_num', type=int, default=-1)
    args = parser.parse_args()
    args.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # cross validation
    if args.all_data == False:
        confusion_matrixes = []
        for i in range(1, args.n_fold+1):
            args.seed_num = i
            # train
            train_cross_validation(args)
            test_loader = DataLoader(MIL(split="test", feat_dir=args.feat_dir, seed_num=args.seed_num, inf=True),
                                    batch_size=1,
                                    num_workers=4,
                                    shuffle=False)
            # inference
            combine_cm = inference_cross_validation(args, test_loader)
            confusion_matrixes.append(combine_cm)
        plot_confusion_matrix(args, np.sum(confusion_matrixes, axis=0), target_names=['Oligo', 'Astro'], normalize=True, title='Confusion Matrix_CV')
    # all data
    else:
        # train
        train_all_data(args)
        test_loader = DataLoader(MIL(split="all", feat_dir=args.feat_dir, inf=True),
                                batch_size=1,
                                num_workers=4,
                                shuffle=False)
        # inference(no bootstrap)
        if args.boot_num < 0:
            inference(args, test_loader) 
        # inference(bootstrap)
        else:
            inference_bootstrap(args, test_loader)

    
