from distutils.archive_util import make_archive
import torch
from torch.utils.data import DataLoader
from data_loader_all import my_collate_fn
from data_loader_all import MIL as MIL
# from model_all import Model
from model_all import Model
import os
import numpy as np
from os.path import join as ospj
from argparse import ArgumentParser
import math
from PIL import Image
import h5py
import pickle
import matplotlib.pyplot as plt
import cv2
import openslide
from openslide import open_slide, ImageSlide, OpenSlide
from openslide.deepzoom import DeepZoomGenerator
from skimage import exposure
import time
import statistics
import shutil
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.utils import make_grid

class visualization:
    def __init__(self, slide, label, center, size):
        self.label = label.item()
        self.short_slide = slide[0]     # S21-00303
        self.slide = self.find_full_slide_name(self.label, self.short_slide)    # SS21-00303-#sdksdksak.svs
        print('self_slide : ', self.slide, 'self_label : ', self.label)
        self.image = open_slide(ospj('/data1/wsi/svs_images', str(self.label), self.slide))
        # find specific location 
        self.center = center
        self.size = size
        
        self.MAG_BASE = self.image.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
        self.level = int(math.log2(float(self.MAG_BASE)/20))  # 사실상 1
        if args.specific_image == True:
            self.full_tile, self.specific_center = self.specific_tile(self.image, self.size, self.level, self.center)

        # find svs file name
    def find_full_slide_name(self, label, slide):
        # print(slide)
        if label == 0:
            slides = os.listdir(ospj('/data1/wsi/svs_images', str(label)))
            for s in slides:
                if slide in s:
                    full_slide_name = s
                    return full_slide_name
        else:
            slides = os.listdir(ospj('/data1/wsi/svs_images', str(label)))
            for s in slides:
                if slide in s:
                    full_slide_name = s
                    return full_slide_name
        
        
        # specific location(size) image
    def specific_tile(self, image, size, level, center):
        
        full_dz =  DeepZoomGenerator(image, tile_size = 224 * size, overlap=0, limit_bounds=False)
        center_col, center_row = center[0], center[1]
        specific_center_col = math.ceil((center_col + 1) / size) - 1
        specific_center_row = math.ceil((center_row + 1) / size) - 1
        specific_center = (specific_center_col, specific_center_row)
        level = full_dz.level_count - level - 1  # 16 = 18 - 1 - 1  (-1은 index start 0)\
        full_tile = full_dz.get_tile(level, specific_center)     # (tile_size)^2
        return full_tile, specific_center
    

    #     # 일부 이미지 Heatmap (visualization)
    # def specific_blending(self, slide_name, slide2pos, center, size):
        
    #     np_full_tile = cv2.cvtColor(np.array(self.full_tile), cv2.COLOR_RGB2BGR)
    #     gray_np_full_tile = cv2.cvtColor(np.array(self.full_tile), cv2.COLOR_RGB2GRAY)
    #     back = np.zeros(gray_np_full_tile.shape, dtype=np_full_tile.dtype)
        
    #     total_tiles = len(slide2pos)
    #     for i in range(total_tiles):    # T
    #         col, row = slide2pos[i]['pos'][0], slide2pos[i]['pos'][1]
    #         if (center[0] - size <= col) and (col <= center[0] + size) and (center[1] - size <= row) and (row <= center[1] + size):
    #             new_center_col = math.ceil((col + 1) / size) - 1
    #             new_center_row = math.ceil((row + 1) / size) - 1
                
    #             if (new_center_col == self.specific_center[0]) and (new_center_row == self.specific_center[1]):
    #                 start_col = col % size  
    #                 start_row = row % size 
    #                 tile_score = slide2pos[i]['score'] 
    #                 for k in range(223):
    #                     for j in range(223):
    #                         color = 255 * tile_score
    #                         # PIL(col, row, 3) --> Numpy (row, col, 3)
    #                         back[(start_row*224) + k][(start_col*224) + j] = color                 
                            
    #     back = cv2.applyColorMap(back, cv2.COLORMAP_JET)
    #     alpha = 0.3
    #     blend = cv2.addWeighted(back, alpha, np_full_tile, 1-alpha, 0)
    #     cv2.imwrite(ospj('specific_tile', slide_name + '_' + str(args.center[0]) + '_' + str(args.center[1]) + '_' + str(args.sizeZZzzZZ) + '.jpeg'), blend)
    #                     #  self.short_slide + '.jpeg'), blend)
        
            # 일부 이미지 Heatmap (visualization)
    def specific_blending(self, slide2pos, which='max'):
        # print(f'{which} score : ')
        np_full_tile = cv2.cvtColor(np.array(self.full_tile), cv2.COLOR_RGB2BGR)
        gray_np_full_tile = cv2.cvtColor(np.array(self.full_tile), cv2.COLOR_RGB2GRAY)
        back = np.zeros(gray_np_full_tile.shape, dtype=np_full_tile.dtype)
        
        total_tiles = len(slide2pos)
        useless_list = [[i,j] for i in range(self.size) for j in range(self.size)]
        # ts_list = []
        for i in range(total_tiles):    # T
            col, row = slide2pos[i]['pos'][0], slide2pos[i]['pos'][1]
            if (self.center[0] - self.size <= col) and (col <= self.center[0] + self.size) and (self.center[1] - self.size <= row) and (row <= self.center[1] + self.size):
                new_center_col = math.ceil((col + 1) / self.size) - 1
                new_center_row = math.ceil((row + 1) / self.size) - 1
                
                if (new_center_col == self.specific_center[0]) and (new_center_row == self.specific_center[1]):
                    start_col = col % self.size  
                    start_row = row % self.size 
                    tile_score = slide2pos[i]['score'] 
                    useless_list = [x for x in useless_list if x != [start_row, start_col]]
                    # loc_list = list(set(loc_list) - set([start_row, start_col]))
                    # ts_list.append([[start_row,start_col], round(tile_score,2)])
                    for k in range(223):
                        for j in range(223):
                            color = 255 * tile_score
                            # PIL(col, row, 3) --> Numpy (row, col, 3)
                            back[(start_row*224) + k][(start_col*224) + j] = color        
        # print(self.label, self.short_slide, ':', ts_list)         
        back = cv2.applyColorMap(back, cv2.COLORMAP_JET)
        
        for [start_row, start_col] in useless_list:
            for k in range(223):
                        for j in range(223):
                            back[(start_row*224) + k][(start_col*224) + j] = 0        
        alpha = 0.6
        blend = cv2.addWeighted(back, alpha, np_full_tile, 1-alpha, 0)
        src = '/home/dhlee/Chowder_Practice/visual/Visualization'
        if which == 'max':
            if self.label == 1:
                # os.makedirs(ospj('specific_tile', args.dname_model.replace('/','_'), 'label_1', self.short_slide, 'max', str(args.size)), exist_ok=True)
                # cv2.imwrite(ospj('specific_tile', args.dname_model.replace('/','_'), 'label_1', self.short_slide, 'max', str(args.size), self.short_slide + '_max_' + str(self.center[0]) + '_' + str(self.center[1]) + '_' + str(self.size) + '.jpeg'), blend)
                # cv2.imwrite(ospj('specific_tile', args.dname_model.replace('/','_'), 'label_1', self.short_slide, 'max', str(args.size), self.short_slide + '_max_' + str(self.center[0]) + '_' + str(self.center[1]) + '_' + str(self.size) + '_origin.jpeg'), np_full_tile)
                
                os.makedirs(ospj(src, 'specific_tile', args.dname_model.replace('/','_'), 'label_1', str(args.size), 'max'), exist_ok=True)
                cv2.imwrite(ospj(src, 'specific_tile', args.dname_model.replace('/','_'), 'label_1', str(args.size), 'max', self.short_slide + '_max_' + str(self.center[0]) + '_' + str(self.center[1]) + '_' + str(self.size) + '.jpeg'), blend)
                cv2.imwrite(ospj(src, 'specific_tile', args.dname_model.replace('/','_'), 'label_1', str(args.size), 'max', self.short_slide + '_max_' + str(self.center[0]) + '_' + str(self.center[1]) + '_' + str(self.size) + '_origin.jpeg'), np_full_tile)
                
            elif self.label == 0:
                # os.makedirs(ospj('specific_tile', args.dname_model.replace('/','_'), 'label_0', self.short_slide, 'max', str(args.size)), exist_ok=True)
                # cv2.imwrite(ospj('specific_tile', args.dname_model.replace('/','_'), 'label_0', self.short_slide, 'max', str(args.size), self.short_slide + '_max_' + str(self.center[0]) + '_' + str(self.center[1]) + '_' + str(self.size) + '.jpeg'), blend)
                # cv2.imwrite(ospj('specific_tile', args.dname_model.replace('/','_'), 'label_0', self.short_slide, 'max', str(args.size), self.short_slide + '_max_' + str(self.center[0]) + '_' + str(self.center[1]) + '_' + str(self.size) + '_origin.jpeg'), np_full_tile)
                os.makedirs(ospj(src, 'specific_tile', args.dname_model.replace('/','_'), 'label_0', str(args.size), 'max'), exist_ok=True)
                cv2.imwrite(ospj(src, 'specific_tile', args.dname_model.replace('/','_'), 'label_0', str(args.size), 'max', self.short_slide + '_max_' + str(self.center[0]) + '_' + str(self.center[1]) + '_' + str(self.size) + '.jpeg'), blend)
                cv2.imwrite(ospj(src, 'specific_tile', args.dname_model.replace('/','_'), 'label_0', str(args.size), 'max', self.short_slide + '_max_' + str(self.center[0]) + '_' + str(self.center[1]) + '_' + str(self.size) + '_origin.jpeg'), np_full_tile)
                
                
        elif which == 'min':
            if self.label == 1:
                os.makedirs(ospj(src, 'specific_tile', args.dname_model.replace('/','_'), 'label_1', str(args.size), 'min'), exist_ok=True)
                cv2.imwrite(ospj(src, 'specific_tile', args.dname_model.replace('/','_'), 'label_1', str(args.size), 'min', self.short_slide + '_min_' + str(self.center[0]) + '_' + str(self.center[1]) + '_' + str(self.size) + '.jpeg'), blend)
                cv2.imwrite(ospj(src, 'specific_tile', args.dname_model.replace('/','_'), 'label_1', str(args.size), 'min', self.short_slide + '_min_' + str(self.center[0]) + '_' + str(self.center[1]) + '_' + str(self.size) + '_origin.jpeg'), np_full_tile)
            elif self.label == 0:
                os.makedirs(ospj(src, 'specific_tile', args.dname_model.replace('/','_'), 'label_0', str(args.size), 'min'), exist_ok=True)
                cv2.imwrite(ospj(src, 'specific_tile', args.dname_model.replace('/','_'), 'label_0', str(args.size), 'min', self.short_slide + '_min_' + str(self.center[0]) + '_' + str(self.center[1]) + '_' + str(self.size) + '.jpeg'), blend)
                cv2.imwrite(ospj(src, 'specific_tile', args.dname_model.replace('/','_'), 'label_0', str(args.size), 'min', self.short_slide + '_min_' + str(self.center[0]) + '_' + str(self.center[1]) + '_' + str(self.size) + '_origin.jpeg'), np_full_tile)
    


           
    # 전체 이미지 Heatmap (visualization)
    def all_blending(self, slide2pos, pixel, bag_label):
        print('hi!')
        # 1. svs 이미지 축소 + PIL로 변형
        image = OpenSlide(ospj('/data1/wsi/svs_images', str(self.label), self.slide))
        # Openslide --> level dimensions[0] == 원본 svs 크기
        # DeepZoom --> 실제 tile == 원본에서 2배 확대  --> 원본/2 (rescale)
        image_length = image.level_dimensions[0]
        print(f'image length : {image_length}')
        rescale_col = math.floor(image_length[0] // 224 / 2) + 1
        rescale_row = math.floor(image_length[1] // 224 / 2) + 1
        print(f'rescal col, row : {rescale_col}, {rescale_row}')
        
        # PIL image 만들기  (pixel == 1 tile 크기)
        rescale_img = image.get_thumbnail((rescale_col*pixel, rescale_row*pixel))
        rescale_col = rescale_img.size[0]
        rescale_row = rescale_img.size[1]
        print(f'rescal col, row : {rescale_col}, {rescale_row}')
        
        
        # 2. 불완벽한 Resize(비율문제) ---> Zero padding  
        #    + Resize와 동일한 shape의 zero numpy 생성
        #    + Opencv -- alpha blending : 오직 Numpy (PIL 안됨)
        rescale_img = np.array(rescale_img)
        print(f'rescale before img shape : {rescale_img.shape}')
        
        # resize(get_thumbnail)하면서 비율때문에 원하는 크기로 resize가 안되고 조금씩 작아지는 case --> zero padding 
        if rescale_col % pixel != 0 :
            pad_pixel = pixel - rescale_col % pixel 
            rescale_img = np.pad(rescale_img, ((0,0),(0,pad_pixel),(0,0)), 'constant', constant_values=0)
        if rescale_row % pixel != 0 :
            pad_pixel = pixel - rescale_row % pixel 
            rescale_img = np.pad(rescale_img, ((0,pad_pixel),(0,0),(0,0)), 'constant', constant_values=0)
        
        
        print(f'rescale after img shape : {rescale_img.shape}')
        np_full_tile = cv2.cvtColor(rescale_img, cv2.COLOR_RGB2BGR)
        
        # save small svs images
        # cv2.imwrite(ospj('Visualization/' + args.dname_model[:-2] + '_' + str(args.pixel), self.short_slide + '_' +  str(args.pixel) + '_label_' + str(bag_label.item())+ '.jpeg'), np_full_tile)
        
        gray_np_full_tile = cv2.cvtColor(rescale_img, cv2.COLOR_RGB2GRAY)        
        back = np.zeros(gray_np_full_tile.shape, dtype= gray_np_full_tile.dtype)
        print(f'back shape : {back.shape}')

        # 3. 모든 tile들의 저장된 location, score에 따라 tile heatmap 생성
        #    heatmap + resize original image 결합 (Alpha blending)
        
        if args.only_max_min == False:
            total_tiles = len(slide2pos)
            for i in range(total_tiles):    # T
                col, row = slide2pos[i]['pos'][0], slide2pos[i]['pos'][1]
                # row, col = slide2pos[i]['pos'][0], slide2pos[i]['pos'][1]
                color = 255 * slide2pos[i]['score']
                for k in range(pixel-1):
                    for j in range(pixel-1):
                        if pixel * row + k >= back.shape[0] or pixel * col + j >= back.shape[1]:
                            print(row, col, k, j )
                        back[pixel * row + k][pixel * col + j] = color
                        
                
            back = cv2.applyColorMap(back, cv2.COLORMAP_JET)
            alpha = 0.3
            blend = cv2.addWeighted(back, alpha, np_full_tile, 1-alpha, 0)
            
            cv2.imwrite(ospj('/home/dhlee/Chowder_Practice/visual/Visualization/' + args.dname_model.replace('/','_') + '_' + str(args.pixel), 'all', self.short_slide + '_' +  str(args.pixel) + '_label_' + str(bag_label.item())+ '.jpeg'), blend)
        else:
            slide2pos_score = sorted(slide2pos, key = lambda x : x['score'], reverse=True)
            slide2pos_score = slide2pos_score[:100] + slide2pos_score[-100:]
            # sc = [round(list(x.values())[1], 2) for x in slide2pos_score]
            # print(sc)
            # return sc
            
            print(self.short_slide)
            total_tiles = len(slide2pos_score)
            for i in range(total_tiles):    # T
                col, row = slide2pos_score[i]['pos'][0], slide2pos_score[i]['pos'][1]
                color = 255 * slide2pos_score[i]['score']
                for k in range(pixel-1):
                    for j in range(pixel-1):
                        back[pixel * row + k][pixel * col + j] = color
                        
            back = cv2.applyColorMap(back, cv2.COLORMAP_JET)
            alpha = 0.3
            blend = cv2.addWeighted(back, alpha, np_full_tile, 1-alpha, 0)
            cv2.imwrite(ospj('/home/dhlee/Chowder_Practice/visual/Visualization/', args.dname_model.replace('/','_') + '_' + str(args.pixel) , 'only_max_min',  'max_min_' + self.short_slide + '_' +  str(args.pixel) + '_label_' + str(bag_label.item())+ '.jpeg'), blend)
            # cv2.imwrite(ospj('Visualization/', args.dname_model[:-2] + '_' + str(args.pixel), 'min_' + self.short_slide + '_' +  str(args.pixel) + '_label_' + str(bag_label.item())+ '.jpeg'), blend)
            # cv2.imwrite(ospj('Visualization/', args.dname_model[:-2] + '_' + str(args.pixel), 'max_' + self.short_slide + '_' +  str(args.pixel) + '_label_' + str(bag_label.item())+ '.jpeg'), blend)

    def max_min_tile_save(self, slide2pos_score, n_tile):
        if n_tile > 100:
            raise ValueError('n_row must be less than 10')
        src = '/data1/wsi/tile_0214'
        target = ospj('/home/dhlee/Chowder_Practice/visual/Visualization', args.dname_model.replace('/','_') + '_' + str(args.pixel), 'max_min_' + str(n_tile) + '_tile')
        os.makedirs(target, exist_ok=True)
        slide2pos_score = slide2pos_score[:n_tile] + slide2pos_score[-n_tile:]
        tiles_path = []
        for i, slide in enumerate(slide2pos_score[:n_tile]):
            col, row = str(slide['pos'][0]), str(slide['pos'][1])
            tile_name = ospj(self.short_slide, self.short_slide + '_' + col + '_' + row)
            tile_path = ospj(src, tile_name + '.jpeg')
            tiles_path.append(tile_path) 
        pil_images = [Image.open(f) for f in tiles_path]
        tensor_images = [transforms.ToTensor()(img) for img in pil_images]
        # print(f'max tile path : {tiles_path}')
        # Stack the images as a grid
        grid = torch.stack(tensor_images)
        grid = make_grid(grid, nrow=10)

        # Save the grid as an image file
        save_image(grid, ospj(target, self.short_slide + '_max_' + str(n_tile) + '.png'))

        tiles_path = []
        for i, slide in enumerate(slide2pos_score[-n_tile:]):
            col, row = str(slide['pos'][0]), str(slide['pos'][1])
            tile_name = ospj(self.short_slide, self.short_slide + '_' + col + '_' + row)
            tile_path = ospj(src, tile_name + '.jpeg')
            tiles_path.append(tile_path) 
        pil_images = [Image.open(f) for f in tiles_path]
        tensor_images = [transforms.ToTensor()(img) for img in pil_images]

        # Stack the images as a grid
        grid = torch.stack(tensor_images)
        grid = make_grid(grid, nrow=10)

        # Save the grid as an image file
        save_image(grid, ospj(target, self.short_slide + '_min_' + str(n_tile) + '.png'))


    # save max & min tile
    # def max_min_tile_save(self, slide2pos_score, tile_num):   
    #     src = '/data1/wsi/tile'
    #     target = ospj('/home/dhlee/Chowder_Practice/visual/Visualization', args.dname_model.replace('/','_') + '_' + str(args.pixel), 'max_min_n_tile', self.short_slide)
    #     os.makedirs(target, exist_ok=True)
    #     slide2pos_score = slide2pos_score[:tile_num] + slide2pos_score[-tile_num:]


    #     for i, slide in enumerate(slide2pos_score[:tile_num]):
    #         col, row = str(slide['pos'][0]), str(slide['pos'][1])
    #         tile_name = ospj(self.short_slide, self.short_slide + '_' + col + '_' + row)
    #         shutil.copy(ospj(src, tile_name + '.jpeg'), ospj(target, 'max' + str(i+1) + '_' + self.short_slide + '_' + col + '_' + row  + '.jpeg'))



    #     for i, slide in enumerate(slide2pos_score[-tile_num:]):
    #         col, row = str(slide['pos'][0]), str(slide['pos'][1])
    #         tile_name = ospj(self.short_slide, self.short_slide + '_' + col + '_' + row)
    #         shutil.copy(ospj(src, tile_name + '.jpeg'), ospj(target, 'min' + str(tile_num - i) + '_' + self.short_slide + '_' + col + '_' + row  + '.jpeg'))
        
    #     # for i in slide2pos_score[-100:]:
    #     #     min_list.append(i['score'])
    #     # print(f'{slide_order} / {slide} ')

        
        


# images score list have different size --> can't use library.....
def normalize(all_score_list, value_reverse=False):
    max, min = 0, 0
    for score_list in all_score_list:
        for i in range(len(score_list)):
            if score_list[i] > max:
                max = score_list[i]
            if score_list[i] < min:
                min = score_list[i]
                
    for score_list in all_score_list:
        for i in range(len(score_list)):
            score_list[i] -= min
            score_list[i] /= (max-min)
            
    if value_reverse == True:
        for x in all_score_list:
            for j in range(len(x)):
                x[j] = 1 - x[j]
    return all_score_list
    
            
@torch.no_grad()
def test(args, test_loader, model):
    
    # find best_model's path
    # best model == best accuracy in validation dataset
    dname_model= ospj(args.dname_model)
    dmodel_path = ospj("/home/dhlee/Chowder_Practice/model3", dname_model)
    print(dmodel_path)
    f_model = os.listdir(dmodel_path)
    model_num = [int(num[11:-4]) for num in f_model]
    best_num = sorted(model_num, reverse=True)[0]
    # load best model
    check_point = torch.load(ospj(dmodel_path,'best_model_' + str(best_num) + '.pth'), map_location='cpu')
    model.load_state_dict(check_point)
    model.eval()
    
    All_score_list = []
    bag_label_dict = {0: [], 1: []}
    print(f'data num : {len(test_loader)}')
    for slide_order, (bag_feature, bag_label, ngs_1, ngs_19, slide) in enumerate(test_loader):
        bag_feature = bag_feature.to(torch.float32).to(args.device)   # (BS x N_tiles x 2048)
        batch_logits, tile_score = model(bag_feature)       # [BS x 2]
        # if slide[0] in temp:
        #     value = batch_logits[0].item()
        #     f.write(f'{slide[0]} : {value}\n')
            # print(slide[0], batch_logits[0].item())
        # Pickle : tile's location(col, row) & Score 
        # with open(ospj('/home/dhlee/Chowder_Practice/feature/col_row_pick', slide[0] + '.pickle'), 'rb') as fr:
        with open(ospj('/home/dhlee/Chowder_Practice/feature', args.data_dir, 'pickle', slide[0] + '.pickle'), 'rb') as fr:
            col_row = pickle.load(fr)
        # slide2pos = []
        score_list = []
        for idx in range(len(col_row)):
            # slide2pos.append(dict(pos=col_row[idx]))
            score_list.append(tile_score[0][idx].item())
        # print(slide)
        # print(bag_label, max(score_list), min(score_list))
        All_score_list.append(score_list)
        if bag_label.item() == 0:
            bag_label_dict[0].append(statistics.mean(score_list))
        else:
            bag_label_dict[1].append(statistics.mean(score_list))
        
    # print(f'0 value : {bag_label_dict[0]} \n\n\n\n\n')
    # print(f'1 value : {bag_label_dict[1]} \n\n\n\n\n')
    # print(f'0 mean : {statistics.mean(bag_label_dict[0])}, 1 mean : {statistics.mean(bag_label_dict[1])}')
    # print(f'0 variance : {statistics.variance(bag_label_dict[0])}, 1 variance : {statistics.variance(bag_label_dict[1])}')
    # print(f'0 max : {max(bag_label_dict[0])}, 1 max : {max(bag_label_dict[1])}')
    # return 0
    if statistics.mean(bag_label_dict[0]) < statistics.mean(bag_label_dict[1]) :
        value_reverse = False
    else:
        value_reverse = True
        
    All_score_list = normalize(All_score_list, value_reverse)
    label_1_score = []
    label_0_score = []

    for slide_order, (bag_feature, bag_label, ngs_1, ngs_19, slide) in enumerate(test_loader):

        temp = ['S18-43964', 'S20-00872', 'S20-64593', 'S21-09644', 'S21-34618', 'S21-41690']
        if slide[0] not in temp:
            continue
        # if slide[0] != 'S19-63682'  and slide[0]!= 'S17-51256' and slide_order > 145:
        # if slide[0] == 'S19-63682'   and slide_order > 100:
        # temp = ['S22-76690', 'S19-66252', 'S21-46229', 'S18-06407', 'S21-70734', 'S21-55597', 'S21-51235', 'S22-26960', 'S17-73140', 'S18-20970', 'S20-32949', 'S17-76411', 'S19-16899', 'S19-83661', 'S22-15174', 'S19-40932', 'S21-09502', 'S19-13580', 'S19-21596', 'S21-45938', 'S21-22664', 'S19-35715', 'S19-51868', 'S20-66423', 'S20-44599', 'S21-56556', 'S18-47458', 'S17-56857', 'S18-12274']
        # if slide[0] != 'S22-65097':
        #     continue
        # if slide[0] not in temp:
        #     continue    
            # bag_feature = bag_feature.to(torch.float32).to(args.device)   # (BS x N_tiles x 2048)
            # batch_logits, tile_score = model(bag_feature)       # [BS x 2]
            
            # Pickle : tile's location(col, row) & Score 
            # with open(ospj('/home/dhlee/Chowder_Practice/feature/col_row_pick', slide[0] + '.pickle'), 'rb') as fr:
        with open(ospj('/home/dhlee/Chowder_Practice/feature', args.data_dir, 'pickle', slide[0] + '.pickle'), 'rb') as fr:
            col_row = pickle.load(fr)
        
        slide2pos = []
        for idx in range(len(col_row)):
            slide2pos.append(dict(pos=col_row[idx]))
        score_list = All_score_list[slide_order]
            # score_list.append(tile_score[0][idx].item())
        # All_score_list.append(score_list)    
        # print(bag_label, max(score_list), min(score_list))
        # continue
        
        # rescale score
        # score_list = exposure.rescale_intensity(score_list, out_range=(0, 1))
        for idx in range(len(col_row)):
            slide2pos[idx]['score'] = score_list[idx]

        
        slide2pos = sorted(slide2pos, key = lambda x : x['pos'][0])
        slide2pos_score = sorted(slide2pos, key = lambda x : x['score'], reverse=True)
        
        # max_list = []
        # min_list = []
        # for i in slide2pos_score[:100]:
        #     max_list.append(i['score'])
        # for i in slide2pos_score[-100:]:
        #     min_list.append(i['score'])
        # print(f'{slide_order} / {slide} ')
        # print('max', max(max_list), min(max_list))
        # print('\n')
        # print('min', max(min_list), min(min_list))
        # print('min', slide2pos_score[-100:])
        if args.specific_image == False:
            # os.makedirs('Visualization/' + args.dname_model.replace('/','_') + '_' + str(args.pixel), exist_ok=True)
            os.makedirs(ospj('/home/dhlee/Chowder_Practice/visual/Visualization/', args.dname_model.replace('/','_') + '_' + str(args.pixel) , 'all'), exist_ok=True)
            os.makedirs(ospj('/home/dhlee/Chowder_Practice/visual/Visualization/', args.dname_model.replace('/','_') + '_' + str(args.pixel) , 'only_max_min'), exist_ok=True)
            visual = visualization(slide, bag_label, args.center, args.size)
            visual.all_blending(slide2pos, 10, bag_label)
            
            # sc = visual.all_blending(slide2pos, 10, bag_label)
            # if bag_label == 1:
            #     label_1_score.append(sc)
            # elif bag_label == 0:
            #     label_0_score.append(sc)
            
        else:
            center = slide2pos_score[0]['pos']
            visual = visualization(slide, bag_label, center, args.size)
            # visual.specific_blending(slide2pos, 'max')
            
            center = slide2pos_score[-1]['pos']
            visual = visualization(slide, bag_label, center, args.size)
            # visual.specific_blending(slide2pos, 'min')

            if args.max_min_tile == True:
                visual.max_min_tile_save(slide2pos_score, args.n_tile)
            
    # label_1_score = sum(label_1_score,[])
    # label_0_score = sum(label_0_score,[])
    # plt.hist(label_1_score, color='r')
    # plt.hist(label_0_score)
    # plt.title('max_min_score')
    # plt.xlabel('label 1 = red')
    # plt.tight_layout()
    # plt.savefig('savefig_default3.png')
            
        
if __name__ == '__main__' :
    parser = ArgumentParser()
    parser.add_argument('--dname_model', type=str, default='models')
    parser.add_argument('--max_r', type=int, default=100)
    parser.add_argument('--mode', type=str, default='Regression')
    parser.add_argument('--infer', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--data_dir', type=str, default='0103_feature')
    parser.add_argument('--loss', type=str, default='one')
    parser.add_argument('--data', type=str, default='test')
    parser.add_argument('--seed_num', type=int, default=1)
    parser.add_argument('--pixel', type=int, default=10)
    parser.add_argument('--only_max_min', action='store_true')
    parser.add_argument('--specific_image', action='store_true')
    parser.add_argument('--max_min_tile', action='store_true')
    parser.add_argument('--n_tile', type=int, default=100)

    parser.add_argument('--slide_name', type=str, default='S18-78680')
    parser.add_argument('--center', type=int, nargs='+', default=(15,15))
    parser.add_argument('--size', type=int, default=10)
    
    args = parser.parse_args()
    args.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    test_loader = DataLoader(MIL(split=args.data, n_time=0, data_dir=args.data_dir, seed_num=args.seed_num, infer=args.infer),
                                batch_size=1,
                                num_workers=2,
                                shuffle=False)
    
    model = Model(args).to(args.device)           
    
    test(args, test_loader, model)
    


# Tile's max_min value & index num --> Find pos(row,col) --> Make heatmap
        # sort_index = torch.argsort(tile_score, dim=1,descending=True) #[BS x T]
        # max_index = sort_index[:][:args.max_r]
        # max_index = sort_index[0][:args.max_r]      # batch_size = 1 고정으로 할 것이기 때문에 가능한 코드
        # min_index = sort_index[0][tile_score.shape[1] - args.max_r:]
        
        # sort_value = torch.gather(tile_score, 1, sort_index)
        # max_value = sort_value[0][:args.max_r]
        # min_value = sort_value[0][tile_score.shape[1] - args.max_r:]
        # pickle == tile's poistion list  [row,col]
        
# def max_specific_tile(self, image):
        
    #     crop_dz =  DeepZoomGenerator(image, tile_size = 224, overlap=0, limit_bounds=False)
    #     crop_col, crop_row = crop_dz.level_tiles[16]  # 7,4 
    #     possible_size = min(crop_col, crop_row)    # 4
    #     possible_center = (crop_col//2, crop_row//2)       # 3,2
        
    #     full_dz = DeepZoomGenerator(image, tile_size = 224 * possible_size, overlap=0, limit_bounds=False)
    #     specific_center_col = math.ceil((possible_center[0] + 1) / possible_size) - 1
    #     specific_center_row = math.ceil((possible_center[1] + 1) / possible_size) - 1
    #     specific_center = (specific_center_col, specific_center_row)
    #     print('specific_center ', specific_center)
    #     level = full_dz.level_count - self.level - 1  # 16 = 18 - 1 - 1  (-1은 index start 0)\
    #     full_tile = full_dz.get_tile(level, specific_center)     # (tile_size)^2
    #     print('full_tile_size ', full_tile.size)
        
        
    #     return full_tile, specific_center, possible_center, possible_size 
    
# def blending(self, slide2pos, size):
    #     # np_full_tile = np.array(self.full_tile)
    #     np_full_tile = cv2.cvtColor(np.array(self.full_tile), cv2.COLOR_RGB2BGR)
    #     # red_block = np.full((224, 224, 3), (0, 0, 255), dtype=np.uint8)
    #     back = np.zeros(np_full_tile.shape, dtype=np_full_tile.dtype)
        
    #     for i in range(args.max_r): # args.max_r
    #         col, row = slide2pos[i]['pos'][0], slide2pos[i]['pos'][1]
    #         new_center_col = math.ceil((col + 1) / size) - 1
    #         new_center_row = math.ceil((row + 1) / size) - 1
    #         # print(new_center_col, new_center_row, 'new')
    #         if (new_center_col == self.specific_center[0]) and (new_center_row == self.specific_center[1]):
    #             start_col = col % size  # 3
    #             start_row = row % size  # 2
    #             max_score = slide2pos[i]['score'] 
    #             for k in range(224):
    #                 for j in range(224):
    #                     # back[(start_col*224) + k][(start_row*224) + j] = (0,0,255)
    #                     back[(start_row*224) + k][(start_col*224) + j] = (0,0,255)
                        
    #             # print(start_col, start_row, 'JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ')
    #             print(col ,row)
    #             print(start_row, start_col, 'JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ')
                
    #     for i in range(args.max_r): # args.max_r
    #         col, row = slide2pos[i+args.max_r]['pos'][0], slide2pos[i+args.max_r]['pos'][1]
    #         new_center_col = math.ceil((col + 1) / size) - 1
    #         new_center_row = math.ceil((row + 1) / size) - 1
    #         # print(new_center_col, new_center_row, 'new')
    #         if (new_center_col == self.specific_center[0]) and (new_center_row == self.specific_center[1]):
    #             start_col = col % size  # 3
    #             start_row = row % size  # 2
    #             min_score = slide2pos[i]['score'] 
    #             for k in range(224):
    #                 for j in range(224):
    #                     # back[(start_col*224) + k][(start_row*224) + j] = (255,0,0)
    #                     back[(start_row*224) + k][(start_col*224) + j] = (255,0,0)
                        
    #             # print(start_col, start_row, 'KKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK')    
    #             print(col ,row)
    #             print(start_row, start_col, 'KKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK') 
    #         alpha = 0.3
    #         blend = cv2.addWeighted(back, alpha, np_full_tile, 1-alpha, 0)
    #         # print(self.short_slide + '_' + str(self.center[0]) + '_' + str(self.center[1]) + '_' + str(self.size) + 'jpeg')
    #         # cv2.imwrite('/ex.jpeg', blend)
    #         cv2.imwrite(ospj('specific_tile_' + args.pixel, self.short_slide + '_' + str(self.center[0]) + '_' + str(self.center[1]) + '_' + str(self.size) + '.jpeg'), blend)
        