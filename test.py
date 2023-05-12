import os
from os.path import join as ospj
import torch
from torch.utils.data import DataLoader
from train.data_loader_all import MIL as MIL
from train.model_all import Model
from helper.load_model import load_best_model
from helper.combine_model import combine_model
from helper.metrics import model_output, perform, Logistic_perform, perform_bootstrap, Logistic_perform_bootstrap
from argparse import ArgumentParser

@torch.no_grad()
def inference_cv(args, test_loader):
    test_loader = DataLoader(MIL(split='test', data_dir=args.data_dir, seed_num=i),
                                        batch_size=1,
                                        num_workers=1,
                                        shuffle=False)
    
    _1pNet, _19qNet = load_best_model(args, order=args.seed_num)
    print('--------------------inference(CV)----------------------------------------')

    output = model_output(args, test_loader, _1pNet, _19qNet)
    _1pNet_auc, _19qNet_auc, _1pNet_metrics, _19qNet_metrics = perform(args, output)
    # test_cor_1, test_cor_19 = correlation(test_model_output)

    logistic_model = Logistic_reg_model(args, args.seed_num, test_loader, _1pNet, _19qNet, cross_val=True)
    log_auc = Logistic_perform(args, output, logistic_model)

    # result = all_auv_specific(test_model_output)
    # test_logistic_auc, test_logistic_fpr, test_logisitic_tpr, cm = logistic_reg_auc(logistic_model, test_model_output)

    # result.extend([test_logistic_auc, test_logistic_fpr, test_logisitic_tpr])
    # print(result)
    # [And_fpr_np, And_tpr_np, And_auc, test_logistic_auc, test_logistic_fpr, test_logisitic_tpr]
    
    # TODO: R2? plot?
    
    # output = predict_value_mean_std(test_model_output, num)

    # plot_pearson_seed(test_model_output, num)

    AUC = [_1pNet_auc, _19qNet_auc, log_auc]
    METRICS = [_1pNet_metrics, _19qNet_metrics]

    make_csv_cv(args, AUC, METRICS)
    # if args.excel == True:
        measure_others(test_model_output, cm, args, num, result)

    return result, cm, output
    # return 0 


if __name__ == '__main__' :
    parser = ArgumentParser()
    parser.add_argument('--dname_1pNet', type=str, default='model_1')
    parser.add_argument('--dname_19qNet', type=str, default='model_19')
    parser.add_argument('--max_r', type=int, default=100)
    parser.add_argument('--gpu', type=int, default=0)
    # parser.add_argument('--mode', type=str, default='Regression')
    parser.add_argument('--data_dir', type=str, default='0725_after')
    parser.add_argument('--n_fold', type=int, default=10)

    # parser.add_argument('--loss', type=str, default='one')
    # parser.add_argument('--ngs_1_19', type=int, default=1)
    # parser.add_argument('--data', type=str, default='test')
    # parser.add_argument('--seed_num', type=int, default=1)
    # parser.add_argument('--excel', action='store_true') 
    # parser.add_argument('--infer', action='store_true')
    
    
    args = parser.parse_args()
    args.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # model_1 = Model(args).to(args.device)           
    # model_19 = Model(args).to(args.device)      
    
    # all_value = {"And_fpr":[], "And_tpr":[], "And_auc":[], "log_fpr":[], 'log_tpr':[], "log_auc":[]}
    


    for i in range(1, args.n_fold+1):
        args.seed_num = i
        inference_cv(args)

    # for i in range(1, 11, 1):
    #     test_loader = DataLoader(MIL(split='test', data_dir=args.data_dir, seed_num=i),
    #                                 batch_size=1,
    #                                 num_workers=1,
    #                                 shuffle=False)
    
    #     result = And_AUC(args, train_loader, test_loader, model_1, model_19, i)
    #     all_value["And_fpr"].append(result[0])
    #     all_value["And_tpr"].append(result[1])
    #     all_value["And_auc"].append(result[2])
    #     all_value["log_fpr"].append(result[3])
    #     all_value["log_tpr"].append(result[4])
    #     all_value["log_auc"].append(result[5])

    # # boot_roc_curve(all_value)
    # boot_logistic(all_value)