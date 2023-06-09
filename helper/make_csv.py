import os
from os.path import join as ospj
import pandas as pd

def make_csv_external(args, AUC, METRICS, bootstrap=False):
    dir_name = ospj('Performance', args.dname_1pNet + '_' + args.dname_19qNet + '_' + args.feat_dir.split('/')[-1])
    os.makedirs(dir_name, exist_ok=True)
    if bootstrap == False:
        file_path = ospj(dir_name, 'Extenral_perform.csv')
    else:
        file_path = ospj(dir_name, 'Extenral_perform_bootstrap.csv')

    if os.path.isfile(file_path):
        os.remove(file_path)
    df = pd.DataFrame(columns = ['Final_AUC', 'Final_accuracy', 'Final_precision', 'Final_recall', 'Final_f1_score', \
                                 '1pNet_AUC', '1pNet_accuracy', '1pNet_precision', '1pNet_recall', '1pNet_f1_score', \
                                 '19qNet_AUC', '19qNet_accuracy', '19qNet_precision', '19qNet_recall', '19qNet_f1_score'])

    Final_result = [AUC[2], METRICS[2][0], METRICS[2][1], METRICS[2][2], METRICS[2][3]]
    Final_result.extend([AUC[0], METRICS[0][0], METRICS[0][1], METRICS[0][2], METRICS[0][3]])
    Final_result.extend([AUC[1], METRICS[1][0], METRICS[1][1], METRICS[1][2], METRICS[1][3]])

    df.loc[0] = Final_result
    df = df.round(3)
    df.to_csv(file_path, index=False)
    print('-------------------External Performance------------------------')
    print(df)

def make_csv_cross_validation(args, AUC, METRICS):
    dir_name = ospj('Performance', args.dname_1pNet + '_' + args.dname_19qNet + '_' + args.feat_dir.split('/')[-1])
    os.makedirs(dir_name, exist_ok=True)
    file_path = ospj(dir_name, 'Cross_validation_perform.csv')

    if args.seed_num == 1 and args.n_fold != 1:
        if os.path.isfile(file_path):
            os.remove(file_path)
        df = pd.DataFrame(columns = ['Final_AUC', 'Final_accuracy', 'Final_precision', 'Final_recall', 'Final_f1_score', \
                                    '1pNet_AUC', '1pNet_accuracy', '1pNet_precision', '1pNet_recall', '1pNet_f1_score', \
                                    '19qNet_AUC', '19qNet_accuracy', '19qNet_precision', '19qNet_recall', '19qNet_f1_score'])
        df.to_csv(file_path, index=False)

    df = pd.read_csv(file_path)
    df.index += 1
    
    Final_result = [AUC[2], METRICS[2][0], METRICS[2][1], METRICS[2][2], METRICS[2][3]]
    Final_result.extend([AUC[0], METRICS[0][0], METRICS[0][1], METRICS[0][2], METRICS[0][3]])
    Final_result.extend([AUC[1], METRICS[1][0], METRICS[1][1], METRICS[1][2], METRICS[1][3]])

    df.loc[df.shape[0] + 1] = Final_result
    df = df.round(3)
    df.to_csv(file_path, index=False)

    if args.seed_num == args.n_fold:
        print('-------------------Cross Validation Performance------------------------')
        print(df)
    