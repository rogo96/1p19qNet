import torch
from torch.utils.data import DataLoader
from train.data_loader_all import MIL 
from train.model_all import Model
from helper.load_model import load_best_model
from helper.combine_model import combine_model
from helper.metrics import model_output, perform, combine_perform, perform_bootstrap, combine_perform_bootstrap, Mean_metrics
from helper.make_csv import make_csv_external, make_csv_cross_validation
from helper.make_result import plot_confusion_matrix, plot_roc_curve
from argparse import ArgumentParser

@torch.no_grad()
def inference(args, test_loader):
    _1pNet, _19qNet = load_best_model(args)
    print('--------------------inference----------------------------------------')
    output = model_output(args, test_loader, _1pNet, _19qNet)
    _1pNet_auc, _19qNet_auc, _1pNet_metrics, _19qNet_metrics = perform(args, output)

    final_model = combine_model(args, None, test_loader, _1pNet, _19qNet, do='load', cross_val=False)
    combine_auc, combine_metrics, combine_cm = combine_perform(args, output, final_model)

    AUC = [_1pNet_auc, _19qNet_auc, combine_auc]
    METRIC = [_1pNet_metrics, _19qNet_metrics, combine_metrics]

    make_csv_external(args, AUC, METRIC)  
    plot_confusion_matrix(args, combine_cm, target_names=['Oligo', 'Astro'], normalize=True, title='Confusion Matrix(external)')

@torch.no_grad()
def inference_bootstrap(args, test_loader):
    _1pNet, _19qNet = load_best_model(args)
    print('--------------------inference(bootstrap)----------------------------------')
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
    
    roc_1pNet = {'fpr': boot_1p[0] , 'tpr': boot_1p[1], 'auc': boot_1p[3]}
    roc_19qNet = {'fpr': boot_19q[0] , 'tpr': boot_19q[1], 'auc': boot_19q[3]}
    roc_combine = {'fpr': boot_combine[0] , 'tpr': boot_combine[1], 'auc': boot_combine[3]}
    plot_roc_curve(args, roc_1pNet, roc_19qNet, roc_combine)

def inference_cross_validation(args, test_loader):
    _1pNet, _19qNet = load_best_model(args, args.seed_num)
    print(f'--------------------inference(cross_validation, order={args.seed_num})----------------------------------')
    output = model_output(args, test_loader, _1pNet, _19qNet)
    _1pNet_auc, _19qNet_auc, _1pNet_metrics, _19qNet_metrics = perform(args, output)

    final_model = combine_model(args, args.seed_num, test_loader, _1pNet, _19qNet, do='load', cross_val=True)
    combine_auc, combine_metrics, combine_cm = combine_perform(args, output, final_model)

    AUC = [_1pNet_auc, _19qNet_auc, combine_auc]
    METRIC = [_1pNet_metrics, _19qNet_metrics, combine_metrics]

    make_csv_cross_validation(args, AUC, METRIC) 
    return combine_cm 

if __name__ == '__main__' :
    parser = ArgumentParser()
    parser.add_argument('--dname_1pNet', type=str, default='1pNet')
    parser.add_argument('--dname_19qNet', type=str, default='19qNet')
    parser.add_argument('--max_r', type=int, default=100)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--feat_dir', type=str, default='Data/Feature')
    parser.add_argument('--boot_num', type=int, default=-1)
    args = parser.parse_args()
    
    test_loader = DataLoader(MIL(split="all", feat_dir=args.feat_dir, inf=True),
                                batch_size=1,
                                num_workers=4,
                                shuffle=False)
    # only inference 
    if args.boot_num < 0:
        inference(args, test_loader)
    # bootstrap inference
    else:
        inference_bootstrap(args, test_loader)
    
    