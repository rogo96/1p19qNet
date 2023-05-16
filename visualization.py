import torch
from torch.utils.data import DataLoader
from train.data_loader_all import MIL
from helper.load_model import load_best_model
import os
from os.path import join as ospj
from argparse import ArgumentParser
import pickle
import openslide
from openslide import open_slide, ImageSlide, OpenSlide
from visual.visual_helper import normalize_score
from visual.visual_class import Visualization

# Heat map visualization                   
@torch.no_grad()
def visualize(args, test_loader, model, dir_name):
    normalized_score = normalize_score(args, test_loader, model)

    for slide_order, (bag_feature, bag_label, slide) in enumerate(test_loader):
        bag_label = str(bag_label.item())

        wsi_list = os.listdir(ospj(args.wsi_dir, bag_label))
        for wsi in wsi_list:
            if wsi.startswith(slide[0]):
                full_slide_name = wsi
                break

        image = open_slide(ospj(args.wsi_dir, bag_label, full_slide_name))
        MAG_BASE = image.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
        if MAG_BASE == None:
            continue

        with open(ospj(ospj(args.feat_dir, 'Tile_position', slide[0] + '.pickle')), 'rb') as fr:
            col_row = pickle.load(fr)
        
        # slide's tile --> [postion(col,row)]
        slide2pos = [{'pos': pos} for pos in col_row]
        score_list = normalized_score[slide_order]

        # slide's tile --> [score]
        for idx in range(len(col_row)):
            slide2pos[idx]['score'] = score_list[idx]
        
        # slide[normalized score]
        for idx, slide2pos_item in enumerate(slide2pos):
            slide2pos[idx]['score'] = score_list[idx]

        # sort by score & position 
        slide2pos = sorted(slide2pos, key = lambda x : x['pos'][0])
        slide2pos_score = sorted(slide2pos, key = lambda x : x['score'], reverse=True)
        print(f'full_slide_name : {full_slide_name}, bag_label : {bag_label}')
        visualization = Visualization(args, slide[0], full_slide_name, bag_label, dir_name)

        if args.n_fold < 0:
            result_path = ospj('Result', args.dname_1pNet + '_' + args.dname_19qNet + '_' + args.feat_dir.split('/')[-1], 'Visualization', 'all', dir_name)
        else:
            result_path = ospj('Result', args.dname_1pNet + '_' + args.dname_19qNet + '_' + args.feat_dir.split('/')[-1], 'Visualization', str(args.seed_num), dir_name)
        os.makedirs(ospj(result_path, str(args.pixel) +  '_all'), exist_ok=True)
        os.makedirs(ospj(result_path, str(args.pixel) +  '_all(max_min)'), exist_ok=True)
        os.makedirs(ospj(result_path, str(args.pixel) +  '_max_tile'), exist_ok=True)
        os.makedirs(ospj(result_path, str(args.pixel) +  '_min_tile'), exist_ok=True)
        # full image    
        visualization.full(args, slide2pos)
        # max & min tile
        visualization.max_min(args, slide2pos_score)
            
        
if __name__ == '__main__' :
    parser = ArgumentParser()
    parser.add_argument('--dname_1pNet', type=str, default='1pNet')
    parser.add_argument('--dname_19qNet', type=str, default='19qNet')
    parser.add_argument('--max_r', type=int, default=100)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--wsi_dir', type=str, default='Data/WSI')
    parser.add_argument('--tile_dir', type=str, default='Data/Tile')
    parser.add_argument('--feat_dir', type=str, default='Data/Feature')
    parser.add_argument('--pixel', type=int, default=10)
    parser.add_argument('--n_fold', type=int, default=-1)

    args = parser.parse_args()
    args.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    test_loader = DataLoader(MIL(split="all", feat_dir=args.feat_dir, inf=True),
                                batch_size=1,
                                num_workers=4,
                                shuffle=False)
    if args.n_fold < 0:
        _1pNet, _19qNet = load_best_model(args)
        print(f'1pNet : {args.dname_1pNet}')
        visualize(args, test_loader, _1pNet, args.dname_1pNet)
        print(f'19qNet : {args.dname_19qNet}')
        visualize(args, test_loader, _19qNet, args.dname_19qNet)
    else:
        for i in range(1, args.n_fold+1):
            args.seed_num = i
            _1pNet, _19qNet = load_best_model(args, args.seed_num)
            print(f'1pNet : {args.dname_1pNet}, order={args.seed_num}')
            visualize(args, test_loader, _1pNet, args.dname_1pNet)
            print(f'19qNet : {args.dname_19qNet}, order={args.seed_num}')
            visualize(args, test_loader, _19qNet, args.dname_19qNet)
 