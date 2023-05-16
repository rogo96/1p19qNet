from os.path import join as ospj
import os
import torch
from train.model_all import Model


def load_best_model(args, order=None):
    args.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model_1 = Model(args).to(args.device)           
    model_19 = Model(args).to(args.device)       
    if order != None:
        # find best_model's path
        dname_1pNet = ospj('model', args.dname_1pNet + '_' + args.dname_19qNet, args.dname_1pNet, str(order))
        dname_19qNet = ospj('model', args.dname_1pNet + '_' + args.dname_19qNet, args.dname_19qNet, str(order))
    else:
        dname_1pNet = ospj('model', args.dname_1pNet + '_' + args.dname_19qNet, args.dname_1pNet, 'all')
        dname_19qNet = ospj('model', args.dname_1pNet + '_' + args.dname_19qNet, args.dname_19qNet, 'all')
        os.makedirs(dname_1pNet, exist_ok=True)
        os.makedirs(dname_19qNet, exist_ok=True)
    
    # best 1p model
    f_model = os.listdir(dname_1pNet)
    model_num = [int(num[11:-4]) for num in f_model]
    best_num = sorted(model_num, reverse=True)[0]
    # load best model
    check_point_1 = torch.load(ospj(dname_1pNet,'best_model_' + str(best_num) + '.pth'), map_location='cpu')
    
    # best 19q model
    f_model = os.listdir(dname_19qNet)
    model_num = [int(num[11:-4]) for num in f_model]
    best_num = sorted(model_num, reverse=True)[0]
    # load best model
    check_point_19 = torch.load(ospj(dname_19qNet,'best_model_' + str(best_num) + '.pth'), map_location='cpu')

    model_1.load_state_dict(check_point_1)
    model_19.load_state_dict(check_point_19)
    
    model_1.eval()
    model_19.eval()

    return model_1, model_19