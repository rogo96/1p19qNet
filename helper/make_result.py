import matplotlib.pyplot as plt
import numpy as np
import itertools
from os.path import join as ospj
import os
from sklearn.metrics import auc, roc_curve

def plot_confusion_matrix(args, cm, target_names=None, cmap=None, normalize=True, labels=True, title='Confusion matrix'):
    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    plt.figure(figsize=(8, 6), facecolor='w', edgecolor='k')
    plt.rcParams['font.size'] = 15
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names)
        plt.yticks(tick_marks, target_names, rotation=90, va='center')
    elif target_names is None:
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ["",""])
        plt.yticks(tick_marks, ["",""])


    if labels:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.3f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black",
                         fontsize=23)
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black",
                        fontsize=23)

    plt.tight_layout()
    plt.ylabel('Ground Truth')
    plt.xlabel('Predicted')
    # plt.show()
    file_path = ospj('Result', args.dname_1pNet + '_' + args.dname_19qNet + '_' + args.feat_dir.split('/')[-1])
    os.makedirs(file_path, exist_ok=True)
    plt.savefig(ospj(file_path, title + '.png'), dpi=300)
    plt.clf()


def plot_roc_curve(args, _1pNet, _19qNet, combine):
    plt.figure()
    # plot 1pNET ROC curve(bootstrap)
    _1p_fprs = _1pNet['fpr']
    _1p_tprs = _1pNet['tpr']
    _1p_aucs = _1pNet['auc']
    boot_num = len(_1p_aucs)

    tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    for i in range(boot_num):
        fpr = _1p_fprs[i]
        tpr = _1p_tprs[i]
    
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    std_auc = np.std(_1p_aucs)
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Bootstrap ROC Curve')
    plt.plot(mean_fpr, mean_tpr,  label=r"1pNET (AUC = %0.3f $\pm$ %0.3f)" % (mean_auc, std_auc), linestyle=':', color='limegreen')
    
    # plot 19qNET ROC curve(bootstrap)
    _19q_fprs = _19qNet['fpr']
    _19q_tprs = _19qNet['tpr']
    _19q_aucs = _19qNet['auc']

    tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    for i in range(boot_num):
        fpr = _19q_fprs[i]
        tpr = _19q_tprs[i]
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    std_auc = np.std(_19q_aucs)
    plt.plot(mean_fpr, mean_tpr, label=r"19qNET (AUC = %0.3f $\pm$ %0.3f)" % (mean_auc, std_auc),  linestyle=':', color='violet') 

    # plot Final ROC curve(bootstrap)
    _f_fprs = combine['fpr']
    _f_tprs = combine['tpr']
    _f_aucs = combine['auc']

    tprs = []
    mean_fpr = np.linspace(0, 1, 100)

    for i in range(boot_num):
        fpr = _f_fprs[i]
        tpr = _f_tprs[i]
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    std_auc = np.std(_f_aucs)
    plt.plot(mean_fpr, mean_tpr, label='Logistic (AUC = {:.3f} $\pm$ {:.3f})'.format(round(mean_auc,3), round(std_auc,3)), color='b')


    alpha = 0.95
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(_f_aucs, p))
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1.0, np.percentile(_f_aucs, p))
    print('%.1f confidence interval %.3f and %.3f' % (alpha*100, lower, upper))

    p = ((1.0-alpha)/2.0) * 100
    lower_tpr = [max(0,x) for x in np.percentile(tprs, p, axis=0)]
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper_tpr = [min(1,x) for x in np.percentile(tprs, p, axis=0)]
    plt.fill_between(mean_fpr, lower_tpr, upper_tpr, color='grey', alpha=0.3, label='95%% CI (AUC = %.3f ~ %.3f)' % (lower, upper))
    plt.tight_layout()
    file_path = ospj('Result', args.dname_1pNet + '_' + args.dname_19qNet + '_' + args.feat_dir.split('/')[-1])
    os.makedirs(file_path, exist_ok=True)
    plt.legend(loc='lower right')
    plt.savefig(ospj(file_path , 'bootstrap_ROC(external).png'), dpi=300)
    plt.clf()