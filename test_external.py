from distutils.archive_util import make_archive
import torch
from torch.utils.data import DataLoader
from inference.data_loader_no_ngs import External_validation
# from data_loader_for_ours import MIL_ours as MIL_ours
# from model_all import Model
from train.model_all import Model
from helper.load_model import load_best_model
from helper.combine_model import combine_model
from helper.metrics import model_output, perform, combine_perform, perform_bootstrap, combine_perform_bootstrap, Mean_metrics
from helper.make_csv import make_csv_external
from helper.make_result import plot_confusion_matrix, plot_roc_curve
import os
from argparse import ArgumentParser
from sklearn.linear_model import LogisticRegression
import h5py
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
import math
import zipfile
import shutil
from pprint import pprint
import seaborn as sns
import scipy.stats as stats
import tqdm

@torch.no_grad()
def inference(args, test_loader):
    _1pNet, _19qNet = load_best_model(args)
    print('--------------------inference----------------------------------------')
    output = model_output(args, test_loader, _1pNet, _19qNet)
    _1pNet_auc, _19qNet_auc, _1pNet_metrics, _19qNet_metrics = perform(args, output)

    final_model = combine_model(args, None, test_loader, _1pNet, _19qNet, do='load', cross_val=False)
    combine_auc, combine_metrics, combine_cm = combine_perform(args, output, final_model)


    # test_Final_auc, test_Final_fpr, test_combineisitic_tpr = Final_reg_auc(final_model, test_model_output)
    # result.extend([test_Final_fpr, test_combineisitic_tpr, test_Final_auc])
    # [fper_1, tper_1, auc_value_1, fper_19, tper_19, auc_value_19, and_fpr, and_tpr, and_auc, test_Final_fpr, test_combineisitic_tpr, test_Final_auc]

    # AUC = [result[2], result[5], result[8], result[11]]
    # AUC = [_1pNet_auc, _19qNet_auc, combine_auc]
    # METRICS = [_1pNet_metrics, _19qNet_metrics]

    AUC = [_1pNet_auc, _19qNet_auc, combine_auc]
    METRIC = [_1pNet_metrics, _19qNet_metrics, combine_metrics]

    make_csv_external(args, AUC, METRIC)  
    plot_confusion_matrix(args, combine_cm, target_names=['Oligo', 'Astro'], normalize=True, title='Confusion Matrix(external)')

@torch.no_grad()
def inference_bootstrap(args, test_loader):
    _1pNet, _19qNet = load_best_model(args)
    print('--------------------inference(bootstrap)----------------------------------------')
    output = model_output(args, test_loader, _1pNet, _19qNet)
    boot_1p, boot_19q = perform_bootstrap(args, output, 35232)
    mean_1pNet_metrics, mean_1pNet_auc = Mean_metrics(boot_1p)
    mean_19qNet_metrics, mean_19qNet_auc = Mean_metrics(boot_19q)

    final_model = combine_model(args, None, test_loader, _1pNet, _19qNet, do='load', cross_val=False)
    boot_combine = combine_perform_bootstrap(args, output, final_model, 35232)
    mean_combine_metrics, mean_combine_auc = Mean_metrics(boot_combine, mode='combine')
    print(f'mean_combine_metrics: {mean_combine_metrics}')

    AUC = [mean_1pNet_auc, mean_19qNet_auc, mean_combine_auc]
    METRIC = [mean_1pNet_metrics, mean_19qNet_metrics, mean_combine_metrics]

    make_csv_external(args, AUC, METRIC, bootstrap=True)
    # FIXME: 통일해서 confusion matrix 만들어야하나?
    roc_1pNet = {'fpr': boot_1p[0] , 'tpr': boot_1p[1], 'auc': boot_1p[3]}
    roc_19qNet = {'fpr': boot_19q[0] , 'tpr': boot_19q[1], 'auc': boot_19q[3]}
    roc_combine = {'fpr': boot_combine[0] , 'tpr': boot_combine[1], 'auc': boot_combine[3]}
    plot_roc_curve(args, roc_1pNet, roc_19qNet, roc_combine)


if __name__ == '__main__' :
    parser = ArgumentParser()
    parser.add_argument('--dname_1pNet', type=str, default='1pNet')
    parser.add_argument('--dname_19qNet', type=str, default='19qNet')
    parser.add_argument('--max_r', type=int, default=100)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--data_dir', type=str, default='tcga_feature')
    parser.add_argument('--boot_num', type=int, default=-1)
    args = parser.parse_args()
    
    test_loader = DataLoader(External_validation(data_dir=args.data_dir),
                                batch_size=1,
                                num_workers=2,
                                shuffle=False)

    # only inference 
    if args.boot_num < 0:
        inference(args, test_loader)
    # bootstrap inference
    else:
        inference_bootstrap(args, test_loader)
    
    