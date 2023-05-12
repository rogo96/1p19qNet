import torch
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_curve, auc, confusion_matrix
import numpy as np
import tqdm
import pandas as pd
import scipy.stats as stats

def model_output(args, test_loader, model_1, model_19):
    GT_CLASS = []
    PREDICT_1 = []
    PREDICT_19 = []
    slide_name = []
    slide_label = []
    for slide_order, (bag_feature, bag_label, slide) in enumerate(test_loader):
        bag_feature = bag_feature.to(torch.float32).to(args.device)   
        BS, T, F = bag_feature.shape
         
        batch_logits_1, tile_score_1 = model_1(bag_feature)      
        batch_logits_19, tile_score_19 = model_19(bag_feature)      
        
        predict_1 = batch_logits_1.squeeze(1)
        predict_19 = batch_logits_19.squeeze(1)
        
        bag_label = bag_label.to(torch.float32).to(args.device)
        
        GT_CLASS.extend(bag_label.cpu().tolist())
        PREDICT_1.extend(predict_1.cpu().tolist())
        PREDICT_19.extend(predict_19.cpu().tolist())
        
        slide_name.append(slide)
        slide_label.append(int(bag_label.cpu().tolist()[0]))
    
    return [GT_CLASS, PREDICT_1, PREDICT_19, slide_name, slide_label]

def Metrics(fper, tper, thresholds, predict, gt_classes, mode):
    best_threshold = thresholds[np.argmax(tper - fper)] # find cutoff point in ROC Curve

    if mode == 'combine':
        pr = [True if i >= 0.5 else False for i in predict]
    else:
        pr = [True if i <= best_threshold else False for i in predict]
    
    acc = accuracy_score(gt_classes, pr)
    precision = precision_score(gt_classes, pr)
    recall = recall_score(gt_classes, pr)
    f1 = f1_score(gt_classes, pr)
    best_metrics = [acc, precision, recall, f1]
    
    cm = confusion_matrix(gt_classes, pr, labels=[0, 1])
    return best_metrics, cm

def perform(args, output):
    gt_classes = output[0]
    predict_1s = output[1]
    predict_19s = output[2]

    fper_1p, tper_1p, thresholds_1p = roc_curve(gt_classes, predict_1s, pos_label=0)
    _1pNet_metrics, _1pNet_cm = Metrics(fper_1p, tper_1p, thresholds_1p, predict_1s, gt_classes, None)
    _1pNet_AUC = auc(fper_1p, tper_1p)

    fper_19q, tper_19q, thresholds_19q = roc_curve(gt_classes, predict_19s, pos_label=0)
    _19qNet_metrics, _19qNet_cm = Metrics(fper_19q, tper_19q, thresholds_19q, predict_19s, gt_classes, None)
    _19qNet_AUC = auc(fper_19q, tper_19q)
    
    ####plot_roc_curve_mix('fig4_tcga', fper_1, tper_1, auc_value_1, fper_19, tper_19, auc_value_19, args.seed_num)
    # plot_predicit_value_ngs(predict_1s, predict_19s, gt_classes, args)
    # plot_pearson(predict_1s, predict_19s, ngs_1s, ngs_19s, gt_classes, args)
    
    return _1pNet_AUC, _19qNet_AUC, _1pNet_metrics, _19qNet_metrics

def combine_perform(args, output, combine_model):
    gt_classes = output[0]
    predict_1s = output[1]
    predict_19s = output[2]

    all_features = pd.DataFrame({'PREDICT_1':predict_1s, 'PREDICT_19':predict_19s})
    all_targets = pd.DataFrame({'GT_CLASS':gt_classes})

    prob = combine_model.predict_proba(all_features)[:,1]

    fpr, tpr, thresholds = roc_curve(all_targets, prob)

    metrics, cm = Metrics(fpr, tpr, thresholds, prob, gt_classes, mode='combine')

    return auc(fpr, tpr), metrics, cm

def perform_bootstrap(args, output, num):
    gt_classes = np.array(output[0]).astype(int)
    predict_1s = np.array(output[1])
    predict_19s = np.array(output[2])

    n_bootstraps = args.boot_num
    boot_fpr_1p, boot_tpr_1p, boot_threshold_1p, boot_auc_1p,  = [], [], [], []
    boot_fpr_19q, boot_tpr_19q, boot_threshold_19q, boot_auc_19q,  = [], [], [], []
    boot_1p_predict, boot_19q_predict = [], []
    boot_1p_gt, boot_19q_gt = [], []

    rng = np.random.RandomState(num)
    for i in tqdm.trange(n_bootstraps):
        indices = rng.random_integers(0, len(gt_classes) - 1, len(gt_classes))
        
        fper_1, tper_1, thresholds_1 = roc_curve(gt_classes[indices], predict_1s[indices], pos_label=0)
        auc_value_1 = auc(fper_1, tper_1)
        fper_19, tper_19, thresholds_19 = roc_curve(gt_classes[indices], predict_19s[indices], pos_label=0)
        auc_value_19 = auc(fper_19, tper_19)

        boot_1p_predict.append(predict_1s[indices])
        boot_19q_predict.append(predict_19s[indices])
        boot_1p_gt.append(gt_classes[indices])
        boot_19q_gt.append(gt_classes[indices])
        boot_fpr_1p.append(fper_1)
        boot_tpr_1p.append(tper_1)
        boot_threshold_1p.append(thresholds_1)
        boot_auc_1p.append(auc_value_1)
        boot_fpr_19q.append(fper_19)
        boot_tpr_19q.append(tper_19)
        boot_threshold_19q.append(thresholds_19)
        boot_auc_19q.append(auc_value_19)

    boot_1p = [boot_fpr_1p, boot_tpr_1p, boot_threshold_1p, boot_auc_1p, boot_1p_predict, boot_1p_gt]
    boot_19q = [boot_fpr_19q, boot_tpr_19q, boot_threshold_19q, boot_auc_19q, boot_19q_predict, boot_19q_gt]


    return boot_1p, boot_19q

def combine_perform_bootstrap(args, output, combine_model, num):
    gt_classes = output[0]
    predict_1s = output[1]
    predict_19s = output[2]

    all_features = pd.DataFrame({'PREDICT_1':predict_1s, 'PREDICT_19':predict_19s})
    all_targets = pd.DataFrame({'GT_CLASS':gt_classes})
    
    n_bootstraps = args.boot_num
    boot_fpr, boot_tpr, boot_threshold, boot_auc, boot_predict, boot_gt = [], [], [], [], [], []


    rng = np.random.RandomState(num)
    for i in tqdm.trange(n_bootstraps):
        indices = rng.random_integers(0, len(gt_classes) - 1, len(gt_classes))
        features = all_features.iloc[indices]
        targets = all_targets.iloc[indices]
        prob = combine_model.predict_proba(features)[:,1]

        fpr, tpr, thresholds = roc_curve(targets,  prob)
        auc_value = auc(fpr, tpr)

        boot_fpr.append(fpr)
        boot_tpr.append(tpr)
        boot_threshold.append(thresholds)
        boot_auc.append(auc_value)
        boot_predict.append(prob)
        boot_gt.append(targets)

    return [boot_fpr, boot_tpr, boot_threshold, boot_auc, boot_predict, boot_gt]

def Mean_metrics(boot_output, mode=None):
    fprs = boot_output[0]
    tprs = boot_output[1]
    thresholds = boot_output[2]
    aucs = boot_output[3]
    predicts = boot_output[4]
    gt_classes = boot_output[5]

    mean_metrics = []
    for i in range(len(aucs)):
        metrics, cm = Metrics(fprs[i], tprs[i], thresholds[i], predicts[i], gt_classes[i], mode)
        mean_metrics.append(metrics)
    
    mean_metrics = np.mean(np.array(mean_metrics), axis=0)
    print(f'len mean metrics, {len(mean_metrics)}')
    mean_auc = np.mean(np.array(aucs))
    
    return mean_metrics, mean_auc