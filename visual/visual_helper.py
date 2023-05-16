import torch
import os
from os.path import join as ospj
import pickle
import statistics
import openslide
from openslide import open_slide, ImageSlide, OpenSlide


def normalize(all_score_list, value_reverse=False):
    all_value = [score for score_list in all_score_list for score in score_list]
    max_val = max(all_value)
    min_val = min(all_value)

    normalized_list = []
    for score_list in all_score_list:
        normalized_scores = [(score - min_val) / (max_val - min_val) for score in score_list]
        normalized_list.append(normalized_scores)

    if value_reverse:
        normalized_list = [[1 - score for score in score_list] for score_list in normalized_list]

    return normalized_list



def normalize_score(args, test_loader, model):
    all_score_list = []
    bag_label_dict = {0: [], 1: []}
    print(f'data num : {len(test_loader)}')

    for slide_order, (bag_feature, bag_label, slide) in enumerate(test_loader):
        bag_feature = bag_feature.to(torch.float32).to(args.device)  
        batch_logits, tile_score = model(bag_feature)     

        # Pickle : tile's location(col, row) & Score 
        tile_position_path = ospj(args.feat_dir, 'Tile_position', slide[0] + '.pickle')
        with open(tile_position_path, 'rb') as fr:
            col_row = pickle.load(fr)

        score_list = [tile_score[0][idx].item() for idx in range(len(col_row))]
        all_score_list.append(score_list)

        bag_label_dict[bag_label.item()].append(statistics.mean(score_list))

    if statistics.mean(bag_label_dict[0]) < statistics.mean(bag_label_dict[1]) :
        value_reverse = False
    else:
        value_reverse = True
        
    normalized_list = normalize(all_score_list, value_reverse)

    return normalized_list

def slide_to_position_score(args, test_loader, normalized_score):
    for slide_order, (bag_feature, bag_label, slide) in enumerate(test_loader):
        image = open_slide(ospj(args.tile_path, str(bag_label), slide[0]+'.h5'))
        MAG_BASE = image.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
        if MAG_BASE == None:
            continue
        
        with open(ospj(args.feat_dir, 'Tile_position', slide[0] + '.pickle'), 'rb') as fr:
            col_row = pickle.load(fr)
        
        # slide[postion]
        slide2pos = [{'pos': pos} for pos in col_row]
        score_list = normalized_score[slide_order]
        
        # slide[normalized score]
        for idx, slide2pos_item in enumerate(slide2pos):
            slide2pos['score'] = score_list[idx]

        slide2pos = sorted(slide2pos, key = lambda x : x['pos'][0])
        slide2pos_score = sorted(slide2pos, key = lambda x : x['score'], reverse=True)

        

