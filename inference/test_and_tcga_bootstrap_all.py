from distutils.archive_util import make_archive
import torch
from torch.utils.data import DataLoader
from data_loader_no_ngs import MIL as MIL
from data_loader_for_ours import MIL_ours as MIL_ours
# from model_all import Model
from model_all import Model
import os
import numpy as np
import pandas as pd
from os.path import join as ospj
from argparse import ArgumentParser
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_curve, auc, confusion_matrix
from sklearn.linear_model import LogisticRegression
import h5py
from logistic_reg import Logistic_reg_model
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
import math
import shutil
from pprint import pprint
import seaborn as sns
import scipy.stats as stats
import tqdm



def boot_all_in_one(result):
    file_path = ospj('/home/dhlee/Chowder_Practice/fig', 'fig4_and_boot')
    print(file_path)
    os.makedirs(file_path, exist_ok=True)
    plt.figure(figsize=(7,6))
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 15
    # plot 1pNET roc curve(bootstrap)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    total_num = 0
    for seed_count in range(len(result['fpr_1'])):
        bootstrap_num = len(result['fpr_1'][seed_count])
        temp_tprs = []
        temp_auc = []
        for i in range(bootstrap_num):
            fper = result['fpr_1'][seed_count][i]
            print(f'fper : {fper}', len(fper))
            tper = result['tpr_1'][seed_count][i]
            auc_value = result['auc_1'][seed_count][i]
        
            interp_tpr = np.interp(mean_fpr, fper, tper)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(auc_value)

            temp_tprs.append(interp_tpr)
            temp_auc.append(auc_value)
    print(f'1 auc : {aucs}, np.std(aucs)')
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    # plt.rcParams['font.family'] = 'Arial'
    # plt.rcParams['font.size'] = 13
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('1000 bootstraps: TCGA(all)')
    # plt.rcParams['font.family'] = 'Arial'
    # plt.rcParams['font.size'] = 15
    plt.plot(mean_fpr, mean_tpr,  label=r"1pNET (AUC = %0.3f $\pm$ %0.3f)" % (mean_auc, std_auc), linestyle=':', color='limegreen')
    
    # plot 19qNET roc curve(bootstrap)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    total_num = 0
    for seed_count in range(len(result['fpr_19'])):
        bootstrap_num = len(result['fpr_19'][seed_count])
        temp_tprs = []
        temp_auc = []
        for i in range(bootstrap_num):
            fper = result['fpr_19'][seed_count][i]
            tper = result['tpr_19'][seed_count][i]
            auc_value = result['auc_19'][seed_count][i]
        
            interp_tpr = np.interp(mean_fpr, fper, tper)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(auc_value)

            temp_tprs.append(interp_tpr)
            temp_auc.append(auc_value)
    print(f'19 auc : {aucs}, {np.std(aucs)}')
        
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    # plt.rcParams['font.family'] = 'Arial'
    # plt.rcParams['font.size'] = 13
    plt.plot(mean_fpr, mean_tpr, label=r"19qNET (AUC = %0.3f $\pm$ %0.3f)" % (mean_auc, std_auc),  linestyle=':', color='violet') 

    

    # all = pd.read_excel("/home/dhlee/Chowder_Practice/excel/1230LDH.xlsx", engine="openpyxl")
    # gt_1 = all[['Class', 'FISH1']].dropna()
    # gt_19 = all[['Class', 'FISH19']].dropna()
    # print(len(gt_1), len(gt_19))
    # fper_1, tper_1, thresholds_1 = roc_curve(gt_1['Class'] ,gt_1['FISH1'], pos_label=0)
    # fper_19, tper_19, thresholds_19 = roc_curve(gt_19['Class'], gt_19['FISH19'], pos_label=0)
    # # And_fpr_np, And_tpr_np, test_auc = And_threshold(thresholds_1, thresholds_19, gt_1['FISH1'], gt_19['FISH19'], gt_1['Class'])
    # plt.plot(fper_19, tper_19, label='FISH_19 = {:.3f} ' .format(round(auc(fper_19, tper_19),3)))
    # plt.legend()
    # plt.tight_layout()



    # plot logistic regression(bootstrap)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 1000)
    
    for seed_count in range(len(result['log_fpr'])):
        bootstrap_num = len(result['log_fpr'][seed_count])
        temp_tprs = []
        temp_aucs = []
        # print(f'result fpr : {result["log_fpr"]}')
        print(len(result['log_fpr'][seed_count])) # bootstrap num
        # for i in range(len(result['log_fpr'][seed_count][0])):
        for i in range(len(result['log_fpr'][seed_count])):

            fper = result['log_fpr'][seed_count][i]
            tper = result['log_tpr'][seed_count][i]
            auc_value = result['log_auc'][seed_count][i]

            interp_tpr = np.interp(mean_fpr, fper, tper)
            interp_tpr[0] = 0.0

            tprs.append(interp_tpr)
            aucs.append(auc_value)

            temp_tprs.append(interp_tpr)
            temp_aucs.append(auc_value)
    print(f'logistic auc : {aucs}, {np.std(aucs)}')

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    # plt.rcParams['font.family'] = 'Arial'
    # plt.rcParams['font.size'] = 13
    plt.plot(mean_fpr, mean_tpr, label='Logistic (AUC = {:.3f} $\pm$ {:.3f})'.format(round(mean_auc,3), round(std_auc,3)), color='b')


    alpha = 0.95
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(aucs, p))
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1.0, np.percentile(aucs, p))
    print('%.1f confidence interval %.3f and %.3f' % (alpha*100, lower, upper))

    p = ((1.0-alpha)/2.0) * 100
    lower_tpr = [max(0,x) for x in np.percentile(tprs, p, axis=0)]
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper_tpr = [min(1,x) for x in np.percentile(tprs, p, axis=0)]
    # plt.rcParams['font.family'] = 'Arial'
    # plt.rcParams['font.size'] = 13
    plt.fill_between(mean_fpr, lower_tpr, upper_tpr, color='grey', alpha=0.3, label='95%% CI (AUC = %.3f ~ %.3f)' % (lower, upper))
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig(ospj(file_path , 'tcga_bootstrap_all_in_on.png'), dpi=300)

    # # FISH logistic
    # all = pd.read_excel(('/home/dhlee/Chowder_Practice/excel/1230LDH.xlsx'), engine='openpyxl')
    # fish_all = all[['Class', 'FISH1', 'FISH19']].dropna()

    # train_features = fish_all[['FISH1', 'FISH19']]
    # train_targets = fish_all[['Class']]

    # logistic_model = LogisticRegression()
    # logistic_model.fit(train_features, train_targets)
    # probs = logistic_model.predict_proba(train_features)[:,1]
    # fper, tper, _ = roc_curve(train_targets, probs)
    # plt.rcParams['font.family'] = 'Arial'
    # plt.rcParams['font.size'] = 13
    # plt.plot(fper, tper, color='red', label='FISH Logistic(AUC = {:.3f})'.format(round(auc(fper, tper),3)))
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(ospj(file_path , 'tcga_bootstrap_all_in_on(fish).png'))  
    # plt.clf()

def roc_curve_figure(result, order):
    file_path = ospj('/home/dhlee/Chowder_Practice/fig', 'fig4_tcga_0321_all')
    os.makedirs(file_path, exist_ok=True)
    # [fper_1, tper_1, auc_value_1, fper_19, tper_19, auc_value_19, and_fpr, and_tpr, and_auc, test_logistic_auc, test_logistic_fpr, test_logisitic_tpr]

    for i in range(0, order):
        
        fper_1 = result[i][0]
        tper_1  = result[i][1]
        auc_value_1 = result[i][2]

        plt.plot(fper_1, tper_1,  label='1p-{} (AUC = {:.3f})'.format(i+1, round(auc_value_1,3)))
        plt.plot([0, 1], [0, 1], color='green', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic Curve(ROC)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(ospj(file_path , 'cum_' + str(i) +  '_1p_roc_.png'))
    plt.clf()
    plt.close()

    for i in range(0, order):
        fper_19 = result[i][3]
        tper_19  = result[i][4]
        auc_value_19 = result[i][5]
        plt.plot(fper_19, tper_19,  label='19q-{} (AUC = {:.3f})'.format(i+1, round(auc_value_19,3)))
        plt.plot([0, 1], [0, 1], color='green', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic Curve(ROC)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(ospj(file_path , 'cum_' + str(i) +  '_19q_roc_.png'))      
    plt.clf()
    plt.close()

    for i in range(0, order):
        fper_and = result[i][6]
        tper_and  = result[i][7]
        auc_value_and = result[i][8]
        plt.plot(fper_and, tper_and,  label='And-{} (AUC = {:.3f})'.format(i+1, round(auc_value_and,3)))
        plt.plot([0, 1], [0, 1], color='green', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic Curve(ROC)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(ospj(file_path , 'cum_' + str(i) +  '_and_roc_.png'))
    plt.clf()
    plt.close()


    for i in range(0, order):
        fper_logistic = result[i][9]
        tper_logistic  = result[i][10]
        auc_value_logistic = result[i][11]
        plt.plot(fper_logistic, tper_logistic,  label='Logistic-{} (AUC = {:.3f})'.format(i+1, round(auc_value_logistic,3)))
        plt.plot([0, 1], [0, 1], color='green', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic Curve(ROC)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(ospj(file_path , 'cum_' + str(i) +  '_logistic_roc_.png'))

    # fper_1 = result[0] 
    # tper_1  = result[1]
    # auc_value_1 = result[2]
    # fper_19 = result[3]
    # tper_19 = result[4]
    # auc_value_19 = result[5]
    

    # plt.plot(fper_1, tper_1,  label='ROC fold 1 (AUC = ' + str(round(auc_value_1,2)) + ')')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic Curve(ROC)')
    # plt.tight_layout()
    # plt.savefig(ospj(file_path , 'cum_' + str(order) +  '_roc_.png'))

    # # tprs = []
    # # aucs = []
    # mean_fpr = np.linspace(0, 1, 100)
    # # total_num = 0
    # # bootstrap_num = len(result[0])
    # for i in range(bootstrap_num):
    #     # total_num = bootstrap_num * seed_count + i + 1
    # #     plt.plot(result[num][0][i], result[num][1][i], color='grey', alpha=0.3)
    #     fper = result[0][i][::-1]
    #     tper = result[1][i][::-1]
    #     auc_value = result[2][i]
    #     # plt.plot(fper, tper, alpha=0.3, label='ROC fold' + str(total_num) +  ' (AUC = ' + str(round(auc_value,2)) + ')')
        
    #     # tprs.append(tper)
    #     # plt.clf()
    #     # plt.show() 
    
    #     interp_tpr = np.interp(mean_fpr, fper, tper)
    #     interp_tpr[0] = 0.0
    #     # print(f'interp_tpr : {interp_tpr}', len(interp_tpr))
    #     tprs.append(interp_tpr)
    #     aucs.append(auc_value)
    
    

    # # plt.savefig(ospj(file_path , str(num) +  '_roc_.png'))
    #     # fprs.append(fper)
    # mean_tpr = np.mean(tprs, axis=0)
    # # print(f'mean tpr : {mean_tpr}')
    # # std_tpr = np.std(tprs, axis=0)
    # # print(f'std_tpr : {std_tpr}')
    # mean_tpr[-1] = 1.0
    # # print(f'mean_tpr : {mean_tpr}', len(mean_tpr))
    # # print(f'mean_fpr : {mean_fpr}', len(mean_fpr))
    # mean_auc = auc(mean_fpr, mean_tpr)
    # std_auc = np.std(aucs)
    
    # plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Mean ROC curve with variability')
    # plt.plot(mean_fpr, mean_tpr, color='blue', label=r"Mean ROC (AUC = %0.3f $\pm$ %0.3f)" % (mean_auc, std_auc)) 
    # # \n(Positive label '{target_names[1]}
    # # plt.legend()
    # # plt.tight_layout()
    
    # # std_tpr = np.std(tprs, axis=0)
    # # tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    # # tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    # alpha = 0.95
    # p = ((1.0-alpha)/2.0) * 100
    # lower_auc = max(0.0, np.percentile(aucs, p))
    # p = (alpha+((1.0-alpha)/2.0)) *100
    # upper_auc = min(1.0, np.percentile(aucs, p))
    # print('auc mean:%.3f, median:%.3f, CI %.3f - %.3f' % (np.mean(aucs), np.median(aucs), lower_auc, upper_auc))
    
    # p = ((1.0-alpha)/2.0) * 100
    # lower_tpr = np.percentile(tprs, p, axis=0)
    # lower_tpr = [max(0,x) for x in lower_tpr]
    # p = (alpha+((1.0-alpha)/2.0)) *100
    # upper_tpr = np.percentile(tprs, p, axis=0)
    # upper_tpr = [min(1.0,x) for x in upper_tpr]
    # # print(f'lower tpr : {lower_tpr}')
    # # print(f'upper_tpr : {upper_tpr}')
    
    # plt.fill_between(mean_fpr, lower_tpr, upper_tpr, color='grey', alpha=0.3)    
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(ospj(file_path , str(num) +  '_roc_.png'))


def boot_roc_curve(result):
    
    file_path = ospj('/home/dhlee/Chowder_Practice/fig', 'fig4_and_boot')
    print(file_path)
    os.makedirs(file_path, exist_ok=True)

    plt.figure(figsize=(6,6))
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 15
    # plot 1pNET roc curve(bootstrap)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    total_num = 0
    for seed_count in range(len(result['fpr_1'])):
        bootstrap_num = len(result['fpr_1'][seed_count])
        temp_tprs = []
        temp_auc = []
        for i in range(bootstrap_num):
            fper = result['fpr_1'][seed_count][i]
            print(f'fper : {fper}', len(fper))
            tper = result['tpr_1'][seed_count][i]
            auc_value = result['auc_1'][seed_count][i]
        
            interp_tpr = np.interp(mean_fpr, fper, tper)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(auc_value)

            temp_tprs.append(interp_tpr)
            temp_auc.append(auc_value)
        # temp_mean_tpr = np.mean(temp_tprs, axis=0)
        # temp_mean_tpr[-1] = 1.0
        # temp_mean_auc = auc(mean_fpr, temp_mean_tpr)
        # temp_std_auc = np.std(temp_mean_auc)
        # print(f'temp_mean_auc : {temp_mean_auc}, temp_std_auc : {temp_std_auc}')
        # plt.plot(mean_fpr, temp_mean_tpr, label='ROC_fold (AUC = {:.3f} $\pm$ {:.3f})'.format(round(temp_mean_auc,3), round(temp_std_auc,3)), alpha=0.3)

        # plt.plot(fper, tper, alpha=0.3, label='ROC fold {} (AUC = {:.3f})'.format(seed_count+1, round(np.mean(temp_auc),3)))
        # plt.plot(fper, tper, alpha=0.3)

    
    mean_tpr = np.mean(tprs, axis=0)
    # print(f'mean tpr : {mean_tpr}')
    # std_tpr = np.std(tprs, axis=0)
    # print(f'std_tpr : {std_tpr}')
    mean_tpr[-1] = 1.0
    # print(f'mean_tpr : {mean_tpr}', len(mean_tpr))
    # print(f'mean_fpr : {mean_fpr}', len(mean_fpr))
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('1000 bootstraps: TCGA(all)')
    # \n(Positive label '{target_names[1]}
    # plt.legend()
    # plt.tight_layout()

    # plt.savefig(ospj(file_path , 'test.png'))
    # plt.plot(mean_fpr, mean_tpr, color='blue', label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc)) 
    plt.plot(mean_fpr, mean_tpr, color='b', label=r"Mean AUC = %0.3f $\pm$ %0.3f" % (mean_auc, std_auc)) 

    
    # std_tpr = np.std(tprs, axis=0)
    # tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    # tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    alpha = 0.95
    p = ((1.0-alpha)/2.0) * 100
    lower_auc = max(0.0, np.percentile(aucs, p))
    p = (alpha+((1.0-alpha)/2.0)) *100
    upper_auc = min(1.0, np.percentile(aucs, p))
    # print('auc mean:%.3f, median:%.3f, CI %.3f - %.3f' % (np.mean(aucs), np.median(aucs), lower_auc, upper_auc))
    print('auc mean:%.3f, median:%.3f, CI %.3f - %.3f' % (mean_auc, np.median(aucs), lower_auc, upper_auc))
    # plt.plot(mean_fpr, mean_tpr, color='blue', label="Mean ROC (AUC = %0.3f ~ %0.3f)" % (lower_auc, upper_auc)) 
    
    p = ((1.0-alpha)/2.0) * 100
    lower_tpr = np.percentile(tprs, p, axis=0)
    lower_tpr = [max(0,x) for x in lower_tpr]
    p = (alpha+((1.0-alpha)/2.0)) *100
    upper_tpr = np.percentile(tprs, p, axis=0)
    upper_tpr = [min(1.0,x) for x in upper_tpr]
    # print(f'lower tpr : {lower_tpr}')
    # print(f'upper_tpr : {upper_tpr}')
    
    plt.fill_between(mean_fpr, lower_tpr, upper_tpr, color='grey', alpha=0.3, label='95%% CI (AUC = %.3f ~ %.3f)' % (lower_auc, upper_auc))
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig(ospj(file_path , 'tcga_bootstrap_1p(all).png'), dpi=300)
    plt.clf()


    ### fish
    # all = pd.read_excel("/home/dhlee/Chowder_Practice/excel/1230LDH.xlsx", engine="openpyxl")
    # gt_1 = all[['Class', 'FISH1']].dropna()
    # gt_19 = all[['Class', 'FISH19']].dropna()
    # print(len(gt_1), len(gt_19))
    # fper_1, tper_1, thresholds_1 = roc_curve(gt_1['Class'] ,gt_1['FISH1'], pos_label=0)
    # fper_19, tper_19, thresholds_19 = roc_curve(gt_19['Class'], gt_19['FISH19'], pos_label=0)
    # # And_fpr_np, And_tpr_np, test_auc = And_threshold(thresholds_1, thresholds_19, gt_1['FISH1'], gt_19['FISH19'], gt_1['Class'])

    # plt.plot(fper_1, tper_1, label='FISH_1 = {:.3f} ' .format(round(auc(fper_1, tper_1),3)))
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(ospj(file_path , 'tcga_bootstrap_1p(all_fish).png'))

    # plt.clf()

    # # FISH logistic
    # all = pd.read_excel(('/home/dhlee/Chowder_Practice/excel/1230LDH.xlsx'), engine='openpyxl')
    # fish_all = all[['Class', 'FISH1', 'FISH19']].dropna()

    # train_features = fish_all[['FISH1', 'FISH19']]
    # train_targets = fish_all[['Class']]

    # logistic_model = LogisticRegression()
    # logistic_model.fit(train_features, train_targets)
    # probs = logistic_model.predict_proba(train_features)[:,1]
    # fper, tper, _ = roc_curve(train_targets, probs)
    # plt.plot(fper, tper, color='red', label='FISH (AUC = {:.3f})'.format(round(auc(fper, tper),3)))
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(ospj(file_path , 'all_with_fish_re_nofish.png')
    
    


    # # bootstrap And_fish
    # fish_tprs = []
    # fish_aucs = []
    # mean_fpr = np.linspace(0, 1, 100)
    # bootstrap_num = 1000
    # gt_class = np.array(gt['Class'].tolist())
    # gt_fish1 = np.array(gt['FISH1'].tolist())
    # gt_fish19 = np.array(gt['FISH19'].tolist())
    # gt_num = len(gt) 
    # rng = np.random.RandomState(10)
    # for i in tqdm.trange(bootstrap_num):
    #     # fper = result[seed_count][0][i][::-1]
    #     # tper = result[seed_count][1][i][::-1]
    #     # auc_value = result[seed_count][2][i]
    #     # plt.plot(fper, tper, alpha=0.3, label='ROC fold' + str(total_num) +  ' (AUC = ' + str(round(auc_value,2)) + ')')
    #     indices = rng.random_integers(0, gt_num - 1, gt_num)
    #     fper_1, tper_1, thresholds_1 = roc_curve(gt_class[indices], gt_fish1[indices], pos_label=0)
    #     fper_19, tper_19, thresholds_19 = roc_curve(gt_class[indices], gt_fish19[indices], pos_label=0)
    #     result = And_threshold(thresholds_1, thresholds_19, gt_fish1[indices], gt_fish19[indices], gt_class[indices])
    #     And_fpr, And_tpr, And_auc_value = result
    #     And_fpr = And_fpr[::-1]
    #     And_tpr = And_tpr[::-1]
    #     # fprs.append(fper)
    #     # tprs.append(tper)
    #     # plt.clf()
    #     # plt.show() 
    
    #     interp_tpr = np.interp(mean_fpr, And_fpr, And_tpr)
    #     interp_tpr[0] = 0.0
    #     # print(f'interp_tpr : {interp_tpr}', len(interp_tpr))
    #     fish_tprs.append(interp_tpr)
    #     fish_aucs.append(And_auc_value)

    # fish_mean_tpr = np.mean(fish_tprs, axis=0)
    # fish_mean_tpr[-1] = 1.0
    # fish_mean_auc = auc(mean_fpr, fish_mean_tpr)
    # fish_std_auc = np.std(fish_aucs)
    # # print(len(fish_tprs), fish_tprs.size())
    # print(f'fish_mean_auc : {fish_mean_auc}, fish_std_auc : {fish_std_auc}')
    # plt.plot(mean_fpr, fish_mean_tpr, label='FISH_boot (AUC = {:.3f} $\pm$ {:.3f})'.format(round(fish_mean_auc,3), round(fish_std_auc,3)), color='red', linestyle=':')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(ospj(file_path , 'all_with_fish_re_boot.png'))
    # # plt.plot(fper, tper, alpha=0.3, label='ROC fold {} (AUC = {:.3f})'.format(seed_count+1, round(np.mean(temp_auc),3)))
    # # plt.plot(fper, tper, alpha=0.3)

    # ### plt.plot(fper_1, tper_1, color='red', label='FISH 1 (AUC = {:.3f})'.format(round(auc(fper_1, tper_1),3)), linestyle='--')
    # ### plt.plot(fper_19, tper_19, color='blue', label='FISH 19 (AUC = {:.d3f})'.format(round(auc(fper_19, tper_19),3)), linestyle='--') 

    # # # plt.savefig(ospj(file_path , 'all_with_fish_and_fish_boot.png'))
    print(file_path)

    plt.figure(figsize=(6,6))
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 15
    # plot 19qNET roc curve(bootstrap)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    total_num = 0
    for seed_count in range(len(result['fpr_19'])):
        bootstrap_num = len(result['fpr_19'][seed_count])
        temp_tprs = []
        temp_auc = []
        for i in range(bootstrap_num):
            fper = result['fpr_19'][seed_count][i]
            tper = result['tpr_19'][seed_count][i]
            auc_value = result['auc_19'][seed_count][i]
        
            interp_tpr = np.interp(mean_fpr, fper, tper)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(auc_value)

            temp_tprs.append(interp_tpr)
            temp_auc.append(auc_value)
        # temp_mean_tpr = np.mean(temp_tprs, axis=0)
        # temp_mean_tpr[-1] = 1.0
        # temp_mean_auc = auc(mean_fpr, temp_mean_tpr)
        # temp_std_auc = np.std(temp_mean_auc)
        # print(f'temp_mean_auc : {temp_mean_auc}, temp_std_auc : {temp_std_auc}')
        # plt.plot(mean_fpr, temp_mean_tpr, label='ROC_fold{} (AUC = {:.3f} $\pm$ {:.3f})'.format(seed_count+1, round(temp_mean_auc,3), round(temp_std_auc,3)), alpha=0.3)

        # plt.plot(fper, tper, alpha=0.3, label='ROC fold {} (AUC = {:.3f})'.format(seed_count+1, round(np.mean(temp_auc),3)))
        # plt.plot(fper, tper, alpha=0.3)

    
    mean_tpr = np.mean(tprs, axis=0)
    # print(f'mean tpr : {mean_tpr}')
    # std_tpr = np.std(tprs, axis=0)
    # print(f'std_tpr : {std_tpr}')
    mean_tpr[-1] = 1.0
    # print(f'mean_tpr : {mean_tpr}', len(mean_tpr))
    # print(f'mean_fpr : {mean_fpr}', len(mean_fpr))
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('1000 bootstraps: TCGA(all)')

    # \n(Positive label '{target_names[1]}
    # plt.legend()
    # plt.tight_layout()

    # plt.savefig(ospj(file_path , 'test.png'))
    # plt.plot(mean_fpr, mean_tpr, color='blue', label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc)) 
    plt.plot(mean_fpr, mean_tpr, color='b', label=r"Mean AUC = %0.3f $\pm$ %0.3f" % (mean_auc, std_auc)) 

    
    # std_tpr = np.std(tprs, axis=0)
    # tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    # tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    alpha = 0.95
    p = ((1.0-alpha)/2.0) * 100
    lower_auc = max(0.0, np.percentile(aucs, p))
    p = (alpha+((1.0-alpha)/2.0)) *100
    upper_auc = min(1.0, np.percentile(aucs, p))
    # print('auc mean:%.3f, median:%.3f, CI %.3f - %.3f' % (np.mean(aucs), np.median(aucs), lower_auc, upper_auc))
    print('auc mean:%.3f, median:%.3f, CI %.3f - %.3f' % (mean_auc, np.median(aucs), lower_auc, upper_auc))
    # plt.plot(mean_fpr, mean_tpr, color='blue', label="Mean ROC (AUC = %0.3f ~ %0.3f)" % (lower_auc, upper_auc)) 
    
    p = ((1.0-alpha)/2.0) * 100
    lower_tpr = np.percentile(tprs, p, axis=0)
    lower_tpr = [max(0,x) for x in lower_tpr]
    p = (alpha+((1.0-alpha)/2.0)) *100
    upper_tpr = np.percentile(tprs, p, axis=0)
    upper_tpr = [min(1.0,x) for x in upper_tpr]
    # print(f'lower tpr : {lower_tpr}')
    # print(f'upper_tpr : {upper_tpr}')
    
    plt.fill_between(mean_fpr, lower_tpr, upper_tpr, color='grey', alpha=0.3, label='95%% CI (AUC = %.3f ~ %.3f)' % (lower_auc, upper_auc))
    plt.legend(fontsize=15)
    plt.tight_layout()
    #tcga_bootstrap_1p(all)
    plt.savefig(ospj(file_path , 'tcga_bootstrap_19q(all).png'), dpi=300)
    plt.clf()

    # # fish
    # all = pd.read_excel("/home/dhlee/Chowder_Practice/excel/1230LDH.xlsx", engine="openpyxl")
    # gt_1 = all[['Class', 'FISH1']].dropna()
    # gt_19 = all[['Class', 'FISH19']].dropna()
    # print(len(gt_1), len(gt_19))
    # fper_1, tper_1, thresholds_1 = roc_curve(gt_1['Class'] ,gt_1['FISH1'], pos_label=0)
    # fper_19, tper_19, thresholds_19 = roc_curve(gt_19['Class'], gt_19['FISH19'], pos_label=0)
    # # And_fpr_np, And_tpr_np, test_auc = And_threshold(thresholds_1, thresholds_19, gt_1['FISH1'], gt_19['FISH19'], gt_1['Class'])
    # plt.plot(fper_19, tper_19, label='FISH_19 = {:.3f} ' .format(round(auc(fper_19, tper_19),3)))
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(ospj(file_path , 'tcga_bootstrap_19q(all_fish).png'))

    # plt.clf()



def boot_logistic(result):
    file_path = ospj('/home/dhlee/Chowder_Practice/fig', 'fig4_logis_boot')
    os.makedirs(file_path, exist_ok=True)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 1000)
    
    plt.figure(figsize=(6,6))
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 15
    for seed_count in range(len(result['log_fpr'])):
        bootstrap_num = len(result['log_fpr'][seed_count])
        temp_tprs = []
        temp_aucs = []
        # print(f'result fpr : {result["log_fpr"]}')
        print(len(result['log_fpr'][seed_count])) # bootstrap num
        # for i in range(len(result['log_fpr'][seed_count][0])):
        for i in range(len(result['log_fpr'][seed_count])):

            fper = result['log_fpr'][seed_count][i]
            tper = result['log_tpr'][seed_count][i]
            auc_value = result['log_auc'][seed_count][i]

            interp_tpr = np.interp(mean_fpr, fper, tper)
            interp_tpr[0] = 0.0

            tprs.append(interp_tpr)
            aucs.append(auc_value)

            temp_tprs.append(interp_tpr)
            temp_aucs.append(auc_value)
        # temp_mean_tpr = np.mean(temp_tprs, axis=0)
        # temp_mean_tpr[-1] = 1.0
        # temp_mean_auc = auc(mean_fpr, temp_mean_tpr)
        # temp_std_auc = np.std(temp_aucs)
        # print(f'temp_logistic_mean_auc : {temp_mean_auc}, temp_logistic_std_auc : {temp_std_auc}')
        # plt.plot(mean_fpr, temp_mean_tpr, label='Logistic_boot{} (AUC = {:.3f} $\pm$ {:.3f})'.format(seed_count+1, round(temp_mean_auc,3), round(temp_std_auc,3)),  linestyle=':', alpha=0.3)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    
    plt.plot([0, 1], [0, 1], linestyle='--', color='green')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('1000 bootstraps : TCGA(all)')

    plt.plot(mean_fpr, mean_tpr, label=r'Mean AUC = {:.3f} $\pm$ {:.3f}'.format(round(mean_auc,3), round(std_auc,3)), color='blue')


    alpha = 0.95
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(aucs, p))
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1.0, np.percentile(aucs, p))
    print('%.1f confidence interval %.3f and %.3f' % (alpha*100, lower, upper))

    p = ((1.0-alpha)/2.0) * 100
    lower_tpr = [max(0,x) for x in np.percentile(tprs, p, axis=0)]
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper_tpr = [min(1,x) for x in np.percentile(tprs, p, axis=0)]

    plt.fill_between(mean_fpr, lower_tpr, upper_tpr, color='grey', alpha=0.3, label='95%% CI (AUC = %.3f ~ %.3f)' % (lower, upper))
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig(ospj(file_path , 'tcga_bootstrap_logis(all).png'), dpi=300)
    plt.clf()


    # # FISH logistic
    # all = pd.read_excel(('/home/dhlee/Chowder_Practice/excel/1230LDH.xlsx'), engine='openpyxl')
    # fish_all = all[['Class', 'FISH1', 'FISH19']].dropna()

    # train_features = fish_all[['FISH1', 'FISH19']]
    # train_targets = fish_all[['Class']]

    # logistic_model = LogisticRegression()
    # logistic_model.fit(train_features, train_targets)
    # probs = logistic_model.predict_proba(train_features)[:,1]
    # fper, tper, _ = roc_curve(train_targets, probs)
    # plt.plot(fper, tper, color='red', label='FISH Logistic(AUC = {:.3f})'.format(round(auc(fper, tper),3)))
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(ospj(file_path , 'tcga_bootstrap_logis(all_fish).png'))  
    # plt.clf()


def plot_roc_curve(dir_name, fper, tper, auc, n_time):
    # plt.subplot(2,1,1)
    file_path = ospj('/home/dhlee/Chowder_Practice/roc_curve', 'logstic_model')
    print(ospj(file_path + '_' + str(n_time) +'.png'))
    os.makedirs(file_path, exist_ok=True)
    plt.plot(fper, tper, color='red', label='AUC = ' + str(round(auc,2)))
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve(ROC)')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(ospj(file_path , str(n_time) +'.png'))
    # plt.show()

def plot_roc_curve_mix(dir_name, fper_1, tper_1, auc_1, fper_19, tper_19, auc_19, n_time):
    # plt.subplot(2,1,1)
    file_path = ospj('/home/dhlee/Chowder_Practice/fig', dir_name)
    print(ospj(file_path + '_' + str(n_time) +'.png'))
    os.makedirs(file_path, exist_ok=True)
    plt.plot(fper_1, tper_1, color='red', label='1pNET AUC = ' + str(round(auc_1,3)))
    plt.plot(fper_19, tper_19, color='blue', label='19qNET AUC = ' + str(round(auc_19,3)))
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve(ROC)')
    plt.legend()
    plt.tight_layout()

    plt.savefig(ospj(file_path , 'stain_norm_mix_seed_' + str(n_time) +'.png'))
    plt.clf()
    # plt.show() 

def plot_roc_curve_2(dir_name, fper, tper, auc, num):
    # plt.subplot(2,1,1)
    file_path = ospj('/home/dhlee/Chowder_Practice/fig', dir_name)
    print(ospj(file_path , str(num) +'.png'))
    os.makedirs(file_path, exist_ok=True)
    plt.plot(fper, tper, color='red', label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve(ROC)')
    # plt.text(0.5, 0.5, 'threshold_ = ' + str(round(threshold,2)), fontsize=10)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(ospj(file_path , 'And_seed_' + str(num) +'.png'))
    plt.clf()
    # plt.show()
    
def make_excel(args, Auc, order):
    os.makedirs('/home/dhlee/Chowder_Practice/And_logis_perform', exist_ok=True)
    dir_name_1 = os.path.dirname(args.dname_model_1)
    dir_name_19 = os.path.dirname(args.dname_model_19)
    dir_name = dir_name_1 + '_tcga_stain_norm_' + dir_name_19
    print('dir name :' , dir_name)
    file_path = ospj('/home/dhlee/Chowder_Practice/And_logis_perform', dir_name + '.csv')
    if not os.path.isfile(file_path):
        df = pd.DataFrame(columns = ['model_1_tcga', 'model_19_tcga', 'And_tcga', 'logistic_tcga'])
        df.to_csv(file_path, index=False)
    
    df = pd.read_csv(file_path)
    df.index += 1
    
    df.loc[order, 'model_1_tcga'] = Auc[0]
    df.loc[order, 'model_19_tcga'] = Auc[1]
    df.loc[order, 'And_tcga'] = Auc[2]
    df.loc[order, 'logistic_tcga'] = Auc[3]
    
    if order == 10:
        df.to_csv(file_path)
    else:
        df.to_csv(file_path, index=False)
        
def model_output(loader, model_1, model_19, args):
    GT_CLASS = []
    PREDICT_1 = []
    PREDICT_19 = []
    slide_name = []
    slide_label = []
    for slide_order, (bag_feature, bag_label, slide) in enumerate(loader):
        bag_feature = bag_feature.to(torch.float32).to(args.device)   # (BS x N_tiles x 2048)
        BS, T, F = bag_feature.shape
         
        batch_logits_1, tile_score_1 = model_1(bag_feature)       # [BS x 2]
        batch_logits_19, tile_score_19 = model_19(bag_feature)       # [BS x 2]
        
        predict_1 = batch_logits_1.squeeze(1)
        predict_19 = batch_logits_19.squeeze(1)
        
        bag_label = bag_label.to(torch.float32).to(args.device)
        
        GT_CLASS.extend(bag_label.cpu().tolist())
        PREDICT_1.extend(predict_1.cpu().tolist())
        PREDICT_19.extend(predict_19.cpu().tolist())
        
        slide_name.append(slide)
        slide_label.append(int(bag_label.cpu().tolist()[0]))
    
    return [GT_CLASS, PREDICT_1, PREDICT_19, slide_name, slide_label]
 

def plot_predicit_value_ngs(predict_1s, predict_19s, gt_classes, args):
    
    class_1_model_1_predict = [predict_1s[i] for i in range(len(gt_classes)) if gt_classes[i] == 1]
    class_0_model_1_predict = [predict_1s[i] for i in range(len(gt_classes)) if gt_classes[i] == 0]
    class_1_ngs_1 = [ngs_1s[i] for i in range(len(gt_classes)) if gt_classes[i] == 1]
    class_0_ngs_1 = [ngs_1s[i] for i in range(len(gt_classes)) if gt_classes[i] == 0]
    
    class_1_model_19_predict = [predict_19s[i] for i in range(len(gt_classes)) if gt_classes[i] == 1]
    class_0_model_19_predict = [predict_19s[i] for i in range(len(gt_classes)) if gt_classes[i] == 0]
    class_1_ngs_19 = [ngs_19s[i] for i in range(len(gt_classes)) if gt_classes[i] == 1]
    class_0_ngs_19 = [ngs_19s[i] for i in range(len(gt_classes)) if gt_classes[i] == 0]
    
    # fig = plt.figure(dpi=300)
    plt.figure()
    plt.scatter(class_1_ngs_1, class_1_model_1_predict, color='r', label='label_1 mean =' + str(round(np.mean(class_1_model_1_predict),2)))
    plt.scatter(class_0_ngs_1, class_0_model_1_predict, color='g', label='label_0 mean =' + str(round(np.mean(class_0_model_1_predict),2)))
    plt.xlabel('NGS_1')
    plt.ylabel('Model_1_predict')
    plt.legend()
    plt.tight_layout()
    plt.savefig(ospj('fig', 'fig3', 'seed_' + str(args.seed_num) + '_ngs_predict_1.png'), dpi=300, facecolor='w')
    plt.clf()
    
    plt.figure()
    plt.scatter(class_1_ngs_19, class_1_model_19_predict, color='r', label='label_1 mean =' + str(round(np.mean(class_1_model_19_predict),2)))
    plt.scatter(class_0_ngs_19, class_0_model_19_predict, color='g', label='label_0 mean =' + str(round(np.mean(class_0_model_19_predict),2)))
    plt.xlabel('NGS_19')
    plt.ylabel('Model_19_predict')
    plt.legend()
    plt.tight_layout()
    plt.savefig(ospj('fig', 'fig3', 'seed_' + str(args.seed_num) + '_ngs_predict_19.png'), dpi=300, facecolor='w')
    plt.clf()
    
def plot_pearson(predict_1s, predict_19s, ngs_1s, ngs_19s, gt_classes, args):
    plt.figure()
    r, p = stats.pearsonr(ngs_1s, predict_1s)
    # print(r,p) # pearson & p-value
    joint = sns.jointplot(ngs_1s, predict_1s, kind="reg")
    joint.set_axis_labels(xlabel='NGS_1', ylabel='Model_1_predict')
    joint.ax_joint.annotate(f'$\\rho = {r:.3f}$', xy=(0.1,0.95), xycoords='axes fraction',\
                        ha='left', va='top', bbox={'boxstyle': 'round', 'fc' : 'none', 'ec' : 'gray'})
    plt.tight_layout()
    plt.savefig(ospj('/home/dhlee/Chowder_Practice/fig', 'fig3', 'corr_seed_' + str(args.seed_num) + '_ngs_predict_1.png'), dpi=300, facecolor='w')
    plt.clf()
    
    plt.figure()
    joint = sns.jointplot(ngs_19s, predict_19s, kind="reg")
    joint.set_axis_labels(xlabel='NGS_19', ylabel='Model_19_predict')
    joint.ax_joint.annotate(f'$\\rho = {r:.3f}$', xy=(0.1,0.95), xycoords='axes fraction',\
                        ha='left', va='top', bbox={'boxstyle': 'round', 'fc' : 'none', 'ec' : 'gray'})
    plt.tight_layout()
    plt.savefig(ospj('/home/dhlee/Chowder_Practice/fig', 'fig3', 'corr_seed_' + str(args.seed_num) + '_ngs_predict_19.png'), dpi=300, facecolor='w')
    plt.clf()
    
def And_threshold_calculation(fpr, tpr, th_1, th_19, and_auc):   
    J = tpr - fpr
    idx = np.argmax(J)
    best_threshold_1 = th_1[idx]
    best_threshold_19 = th_19[idx]
    print('And best threshold 1: ', best_threshold_1)
    print('And best threshold 19: ', best_threshold_19)
    return best_threshold_1, best_threshold_19

def par_metrics(fper, tper, thresholds, predict, gt_classes, num):
    J = tper - fper
    idx = np.argmax(J)
    best_threshold = thresholds[idx]
    pr = [True if i <= best_threshold else False for i in predict]
    
    acc = accuracy_score(gt_classes, pr)
    precision = precision_score(gt_classes, pr)
    recall = recall_score(gt_classes, pr)
    f1 = f1_score(gt_classes, pr)
    best_metrics = [acc, precision, recall, f1]
    
    print(f'par best th = {best_threshold}')
    print(f'par best accuracy: {acc:.3f}, precision: {precision:.3f}, recall: {recall:.3f}, f1: {f1:.3f}')
    
    print(f'par th fix = 0.8')
    pr = [True if i <= 0.8 else False for i in predict]
    acc = accuracy_score(gt_classes, pr)
    precision = precision_score(gt_classes, pr)
    recall = recall_score(gt_classes, pr)
    f1 = f1_score(gt_classes, pr)
    print(f'par th fix accuracy: {acc:.3f}, precision: {precision:.3f}, recall: {recall:.3f}, f1: {f1:.3f}')
    fix_metrics = [acc, precision, recall, f1]
    
    return best_metrics, fix_metrics
    
def And_metrics(predict_1s, predict_19s, gt_classes, threshold_1, threshold_19, num):
    pr_1 = [True if i <= threshold_1 else False for i in predict_1s]
    pr_19 = [True if i <= threshold_19 else False for i in predict_19s]
    
    and_predict = []
    for idx in range(len(pr_1)):
        if pr_1[idx] == True and pr_19[idx] == True:
            and_predict.append(True)
        else:
            and_predict.append(False)
    and_acc = accuracy_score(gt_classes, and_predict)
    and_precision = precision_score(gt_classes, and_predict)
    and_recall = recall_score(gt_classes, and_predict)
    and_f1 = f1_score(gt_classes, and_predict)
    print(f'And best th_1 = {threshold_1}, th_19 = {threshold_19}')
    print(f'And Best accuracy: {and_acc:.3f}, precision: {and_precision:.3f}, recall: {and_recall:.3f}, f1: {and_f1:.3f}')
    best_metrics = [and_acc, and_precision, and_recall, and_f1]
    
    
    print(f'And th fix = 0.8 ')
    threshold_1 = 0.8
    threshold_19 = 0.8
    r_1 = [True if i <= threshold_1 else False for i in predict_1s]
    pr_19 = [True if i <= threshold_19 else False for i in predict_19s]
    
    and_predict = []
    for idx in range(len(pr_1)):
        if pr_1[idx] == True and pr_19[idx] == True:
            and_predict.append(True)
        else:
            and_predict.append(False)
    and_acc = accuracy_score(gt_classes, and_predict)
    and_precision = precision_score(gt_classes, and_predict)
    and_recall = recall_score(gt_classes, and_predict)
    and_f1 = f1_score(gt_classes, and_predict)
    print(f'And th fix accuracy: {and_acc:.3f}, precision: {and_precision:.3f}, recall: {and_recall:.3f}, f1: {and_f1:.3f}')
    fix_metrics = [and_acc, and_precision, and_recall, and_f1]
    
    return best_metrics, fix_metrics
    
def train_val_And_threshold(th_1, th_19, predict_1s, predict_19s, GT):
    
    And_fpr_list = [1]
    And_tpr_list = [1]
    for threshold_1 in th_1[1:]:
        temp_predict_1 = [False if i >= threshold_1 else True for i in predict_1s]
        for threshold_19 in th_19[1:]:
            temp_predict_19 = [False if i >= threshold_19 else True for i in predict_19s]
            And_predict = []
            for x in range(len(temp_predict_1)):
                if temp_predict_1[x] == True and temp_predict_19[x] == True:
                    And_predict.append(True)
                else:
                    And_predict.append(False) 
                    
            tn, fp, fn, tp = confusion_matrix(GT, And_predict).ravel()      
            fpr =  fp /(tn+fp)
            tpr = tp /(tp+fn)
            
            And_fpr_list.append(fpr)    
            And_tpr_list.append(tpr)
    And_fpr_list.append(0)    
    And_tpr_list.append(0)
    And_fpr_np = np.array(And_fpr_list)
    And_tpr_np = np.array(And_tpr_list)
    sort_idx = np.argsort(And_fpr_np)
    And_fpr_np = And_fpr_np[sort_idx[::-1]]
    And_tpr_np = And_tpr_np[sort_idx[::-1]]
    
    And_fpr_np, And_tpr_np = fpr_tpr_sort(And_fpr_np, And_tpr_np)
    
    And_auc = auc(And_fpr_np, And_tpr_np)
    print(f'And auc : {And_auc}')
    # plot_roc_curve_2('fig4_and', And_fpr_np, And_tpr_np, test_auc, args.seed_num)
    return And_auc
    
      
def train_val_par_model_roc(model_output):
    # roc curve * auc(model 1, 19)
    gt_classes = model_output[0]
    predict_1s = model_output[3]
    predict_19s = model_output[4]
    fper, tper, thresholds_1 = roc_curve(gt_classes, predict_1s, pos_label=0)
    auc_value_1 = auc(fper, tper)
    fper, tper, thresholds_19 = roc_curve(gt_classes, predict_19s, pos_label=0)
    auc_value_19 = auc(fper, tper)
    
    # thresholds_1 = [0.9,0.8,0.7,0.6,0.5]
    # thresholds_19 = [0.9,0.8,0.7,0.6,0.5]
    
    # fix_1_auc, fix_19_auc = fix_one_threshold(thresholds_1, thresholds_19, predict_1s, predict_19s, gt_classes)
    and_auc = train_val_And_threshold(thresholds_1, thresholds_19, predict_1s, predict_19s, gt_classes)
    return auc_value_1, auc_value_19, and_auc

def fpr_tpr_sort(fpr, tpr):
    if len(fpr) == len(tpr):
        print('fpr, tpr same num of elements')
    next_idx = 1
    for idx in range(len(fpr)):
        if not next_idx == idx :
            continue
        next_idx = idx + 1 
        while next_idx < len(fpr) and fpr[next_idx] == fpr[idx] :
            next_idx += 1
        if next_idx != idx + 1:
            min = np.min(tpr[idx:next_idx])
            max = np.max(tpr[idx:next_idx])
            tpr[idx] = max
            tpr[idx+1:next_idx] = [max for i in range(len(tpr[idx+1:next_idx]))]
    
    tpr = np.append(tpr, 0)
    fpr = np.append(fpr, 0)
    return fpr, tpr
    
def And_threshold(th_1, th_19, predict_1s, predict_19s, GT):
    
    And_fpr_list = [1]
    And_tpr_list = [1]
    threshold_1_list = [1]
    threshold_19_list = [1]
    for threshold_1 in th_1[1:]:
        temp_predict_1 = [False if i >= threshold_1 else True for i in predict_1s]
        for threshold_19 in th_19[1:]:
            temp_predict_19 = [False if i >= threshold_19 else True for i in predict_19s]
            And_predict = []
            for x in range(len(temp_predict_1)):
                if temp_predict_1[x] == True and temp_predict_19[x] == True:
                    And_predict.append(True)
                else:
                    And_predict.append(False) 
                    
            tn, fp, fn, tp = confusion_matrix(GT, And_predict).ravel()      
            fpr =  fp /(tn+fp)
            tpr = tp /(tp+fn)
            
            And_fpr_list.append(fpr)    
            And_tpr_list.append(tpr)
            threshold_1_list.append(threshold_1)
            threshold_19_list.append(threshold_19)
    And_fpr_list.append(0)    
    And_tpr_list.append(0)
    threshold_1_list.append(0)
    threshold_19_list.append(0)
    And_fpr_np = np.array(And_fpr_list)
    And_tpr_np = np.array(And_tpr_list)
    threshold_1_np = np.array(threshold_1_list)
    threshold_19_np = np.array(threshold_19_list)
    sort_idx = np.argsort(And_fpr_np)
    And_fpr_np = And_fpr_np[sort_idx[::-1]]
    And_tpr_np = And_tpr_np[sort_idx[::-1]]
    threshold_1_np = threshold_1_np[sort_idx[::-1]]
    threshold_19_np = threshold_19_np[sort_idx[::-1]]
    And_fpr_np, And_tpr_np = fpr_tpr_sort(And_fpr_np, And_tpr_np)
    # print(f'And_fpr_np : {And_fpr_np} ', len(And_fpr_np))
    # print(f'And_tpr_np : {And_tpr_np} ', len(And_tpr_np))
    threshold_1_np = np.append(threshold_1_np, 0)
    threshold_19_np = np.append(threshold_19_np, 0)
    test_auc = auc(And_fpr_np, And_tpr_np)
    # print(f'And_fpr : {And_fpr_np} ', len(And_fpr_np))
    # print(f'set fpr : {set(And_fpr_np)} ', len(set(And_fpr_np)))
    # print(f'And_tpr : {And_tpr_np} ', len(And_tpr_np))
    # print(f'And auc : {test_auc}')
    # print(f'set tpr : {set(And_tpr_np)} ', len(set(And_tpr_np)))
    # best_threshold_1, best_threshold_19 = And_threshold_calculation(And_fpr_np, And_tpr_np, threshold_1_np, threshold_19_np, test_auc)
    # plot_roc_curve_2('fig4_tcga', And_fpr_np, And_tpr_np, test_auc, args.seed_num)

    return [And_fpr_np, And_tpr_np, test_auc]
    

def test_par_model_roc(model_output, args, order):
    # roc curve * auc(model 1, 19)
    gt_classes = model_output[0]
    predict_1s = model_output[1]
    predict_19s = model_output[2]
    fper_1, tper_1, thresholds_1 = roc_curve(gt_classes, predict_1s, pos_label=0)
    auc_value_1 = auc(fper_1, tper_1)
    print('auc 1 : ', auc_value_1)
    fper_19, tper_19, thresholds_19 = roc_curve(gt_classes, predict_19s, pos_label=0)
    auc_value_19 = auc(fper_19, tper_19)
    print('auc 19 : ', auc_value_19)

    and_fpr, and_tpr, and_auc = And_threshold(thresholds_1, thresholds_19, predict_1s, predict_19s, gt_classes)


    return [fper_1, tper_1, auc_value_1, fper_19, tper_19, auc_value_19, and_fpr, and_tpr, and_auc] 

def test_par_model_roc_boot(test_model_output, args, num):
    gt_classes = np.array(test_model_output[0]).astype(int)
    predict_1s = np.array(test_model_output[1])
    predict_19s = np.array(test_model_output[2])
    
    n_bootstraps = 1000
    bootstrapped_fpr_1, bootstrapped_tpr_1, bootstrapped_auc_1 = [], [], []
    bootstrapped_fpr_19, bootstrapped_tpr_19, bootstrapped_auc_19 = [], [], []

    rng = np.random.RandomState(num)
    for i in tqdm.trange(n_bootstraps):
        indices = rng.random_integers(0, len(gt_classes) - 1, len(gt_classes))
        fper_1, tper_1, thresholds_1 = roc_curve(gt_classes[indices], predict_1s[indices], pos_label=0)
        auc_value_1 = auc(fper_1, tper_1)
        fper_19, tper_19, thresholds_19 = roc_curve(gt_classes[indices], predict_19s[indices], pos_label=0)
        auc_value_19 = auc(fper_19, tper_19)
        # result = And_threshold(thresholds_1, thresholds_19, predict_1s[indices], predict_19s[indices], gt_classes[indices])

        bootstrapped_fpr_1.append(fper_1)
        bootstrapped_tpr_1.append(tper_1)
        bootstrapped_auc_1.append(auc_value_1)
        bootstrapped_fpr_19.append(fper_19)
        bootstrapped_tpr_19.append(tper_19)
        bootstrapped_auc_19.append(auc_value_19)


    return bootstrapped_fpr_1, bootstrapped_tpr_1, bootstrapped_auc_1, bootstrapped_fpr_19, bootstrapped_tpr_19, bootstrapped_auc_19



@torch.no_grad()
def And_AUC(args, train_loader, test_loader, model_1, model_19, order):
    # find best_model's path
    # best model == best accuracy in validation dataset
    dname_model_1 = ospj(args.dname_model_1, str(order))
    dname_model_19 = ospj(args.dname_model_19, str(order))
    # dname_model_1 = ospj(args.dname_model_1)
    # dname_model_19 = ospj(args.dname_model_19)
    
    dmodel_path_1 = ospj("/home/dhlee/Chowder_Practice/model3", dname_model_1)
    dmodel_path_19 = ospj("/home/dhlee/Chowder_Practice/model3", dname_model_19)
    
    # dmodel_path_1 = ospj("/home/dhlee/Chowder_Practice/ckpt3", dname_model_1)
    # dmodel_path_19 = ospj("/home/dhlee/Chowder_Practice/ckpt3", dname_model_19)
    
    # best 1 model
    f_model = os.listdir(dmodel_path_1)
    model_num = [int(num[11:-4]) for num in f_model]
    best_num = sorted(model_num, reverse=True)[0]
    # load best model
    check_point_1 = torch.load(ospj(dmodel_path_1,'best_model_' + str(best_num) + '.pth'), map_location='cpu')
    
    # best 19 model
    f_model = os.listdir(dmodel_path_19)
    model_num = [int(num[11:-4]) for num in f_model]
    best_num = sorted(model_num, reverse=True)[0]
    # load best model
    check_point_19 = torch.load(ospj(dmodel_path_19,'best_model_' + str(best_num) + '.pth'), map_location='cpu')

    model_1.load_state_dict(check_point_1)
    model_19.load_state_dict(check_point_19)
    
    model_1.eval()
    model_19.eval()
    
    print('--------------------test----------------------------------------')
    test_model_output = model_output(test_loader, model_1, model_19, args)
    logistic_model = Logistic_reg_model(args, train_loader, model_1, model_19, do='load')
    log_fpr, log_tpr, log_auc = logistic_par_model_roc(args, test_model_output, logistic_model, order)
    fpr_1, tpr_1, auc_1, fpr_19, tpr_19, auc_19 = test_par_model_roc_boot(test_model_output, args, order)


    # test_logistic_auc, test_logistic_fpr, test_logisitic_tpr = logistic_reg_auc(logistic_model, test_model_output)
    # result.extend([test_logistic_fpr, test_logisitic_tpr, test_logistic_auc])
    # [fper_1, tper_1, auc_value_1, fper_19, tper_19, auc_value_19, and_fpr, and_tpr, and_auc, test_logistic_fpr, test_logisitic_tpr, test_logistic_auc]

    # AUC = [result[2], result[5], result[8], result[11]]


    # if args.excel == True:
    #     make_excel(args, AUC, order)  

    return [fpr_1, tpr_1, auc_1, fpr_19, tpr_19, auc_19, log_fpr, log_tpr, log_auc]

def logistic_par_model_roc(args, test_model_output, logistic_model, num):
    gt_classes = test_model_output[0]
    predict_1s = test_model_output[1]
    predict_19s = test_model_output[2]

    all_features = pd.DataFrame({'PREDICT_1':predict_1s, 'PREDICT_19':predict_19s})
    all_targets = pd.DataFrame({'GT_CLASS':gt_classes})
    
    n_bootstraps = 1000
    bootstrapped_fpr, bootstrapped_tpr, bootstrapped_auc = [], [], []

    rng = np.random.RandomState(num)
    for i in tqdm.trange(n_bootstraps):
        indices = rng.random_integers(0, len(gt_classes) - 1, len(gt_classes))
        features = all_features.iloc[indices]
        targets = all_targets.iloc[indices]
        prob = logistic_model.predict_proba(features)[:,1]

        fpr, tpr, thresholds = roc_curve(targets,  prob)
        auc_value = auc(fpr, tpr)

        bootstrapped_fpr.append(fpr)
        bootstrapped_tpr.append(tpr)
        bootstrapped_auc.append(auc_value)

    return bootstrapped_fpr, bootstrapped_tpr, bootstrapped_auc


def logistic_reg_auc(logistic_model, model_output):
    gt_classes = model_output[0]
    predict_1s = model_output[1]
    predict_19s = model_output[2]
    
    
    temp_pd = pd.DataFrame({'GT_CLASS':gt_classes, 'PREDICT_1':predict_1s, 'PREDICT_19':predict_19s})
    features = temp_pd[['PREDICT_1', 'PREDICT_19']]
    targets = temp_pd['GT_CLASS']
    
    
    score = logistic_model.score(features, targets)
    print(f'logistic regression score : {score} ')    
    
    prob = logistic_model.predict_proba(features)[:,1]
    
    fpr, tpr, thresholds = roc_curve(gt_classes,  prob)
    print(f'logistic thresholds : {thresholds}')
    
    auc_value = auc(fpr, tpr)
    print(f'logistic auc : {auc_value}')
    # plot_roc_curve('test', fpr, tpr, auc_value, args.seed_num)

     

    return auc_value, fpr, tpr

    
        
            
if __name__ == '__main__' :
    parser = ArgumentParser()
    parser.add_argument('--dname_model_1', type=str, default='model_1')
    parser.add_argument('--dname_model_19', type=str, default='model_19')
    parser.add_argument('--max_r', type=int, default=100)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--mode', type=str, default='Regression')
    parser.add_argument('--data_dir_ours', type=str, default='0214_feature')
    parser.add_argument('--data_dir', type=str, default='tcga_feature')
    parser.add_argument('--loss', type=str, default='one')
    # parser.add_argument('--seed_num', type=int, default=1)
    parser.add_argument('--excel', action='store_true')
    parser.add_argument('--infer', action='store_true')
    
    
    args = parser.parse_args()
    args.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # train_loader = DataLoader(MIL(split='train', n_time=0, data_dir=args.data_dir, seed_num=args.seed_num, infer=args.infer),
    #                             batch_size=1,
    #                             num_workers=2,
    #                             shuffle=False)
    # val_loader = DataLoader(MIL(split='val', n_time=0, data_dir=args.data_dir, seed_num=args.seed_num, infer=args.infer),
    #                             batch_size=1,
    #                             num_workers=2,
    #                             shuffle=False)
    
    model_1 = Model(args).to(args.device)           
    model_19 = Model(args).to(args.device)           
    
    # all_value = {"And_fpr":[], "And_tpr":[], "And_auc":[], "log_fpr":[], 'log_tpr':[], "log_auc":[]}
    all_value = {"fpr_1":[], "tpr_1":[], "auc_1":[], "fpr_19":[], 'tpr_19':[], "auc_19":[], "log_fpr":[], 'log_tpr':[], "log_auc":[]}

    for i in range(1, 2, 1):
        train_loader = DataLoader(MIL_ours(split='all', n_time=0, data_dir=args.data_dir_ours, seed_num=i, infer=args.infer),
                                batch_size=1,
                                num_workers=2,
                                shuffle=False)

        test_loader = DataLoader(MIL(split='test', n_time=0, data_dir=args.data_dir, seed_num=i, infer=args.infer),
                                    batch_size=1,
                                    num_workers=2,
                                    shuffle=False)
        result = And_AUC(args, train_loader, test_loader, model_1, model_19, i)

        all_value["fpr_1"].append(result[0])
        all_value["tpr_1"].append(result[1])
        all_value["auc_1"].append(result[2])
        all_value["fpr_19"].append(result[3])
        all_value["tpr_19"].append(result[4])
        all_value["auc_19"].append(result[5])
        all_value["log_fpr"].append(result[6])
        all_value["log_tpr"].append(result[7])
        all_value["log_auc"].append(result[8])



    # roc_curve_figure(all_value, i)
    boot_roc_curve(all_value)
    boot_logistic(all_value)
    boot_all_in_one(all_value)
    
