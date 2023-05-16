import os
import openslide
from openslide import open_slide, ImageSlide, OpenSlide
from openslide.deepzoom import DeepZoomGenerator
from os.path import join as ospj
import numpy as np
import cv2
import math
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from PIL import Image
import torch

class Visualization:
    def __init__(self, args, slide, full_slide, label, dir_name):
        self.slide = slide      # short slide name (pickle, excel file's name)
        self.full_slide = full_slide  # long slide name (original WSI's name)
        self.label = label
        self.image = open_slide(ospj(args.wsi_dir, self.label, self.full_slide))
        self.pixel = args.pixel
        if args.n_fold < 0:
            self.result_path = ospj('Result', args.dname_1pNet + '_' + args.dname_19qNet + '_' + args.feat_dir.split('/')[-1], 'Visualization', 'all', dir_name)
        else:
            self.result_path = ospj('Result', args.dname_1pNet + '_' + args.dname_19qNet + '_' + args.feat_dir.split('/')[-1], 'Visualization', str(args.seed_num), dir_name)

    # Heatmap of full_WSI 
    def full(self, args, slide2pos):
        # Resizing and Transforming SVS Image using PIL
        image = OpenSlide(ospj(args.wsi_dir, str(self.label), self.full_slide))
        image_length = image.level_dimensions[0]
        rescale_col = math.floor(image_length[0] // 224 / 2) + 1
        rescale_row = math.floor(image_length[1] // 224 / 2) + 1
        
        # Creating a PIL image (where pixel == 1 represents the size of a single tile)
        rescale_img = image.get_thumbnail((rescale_col*self.pixel, rescale_row*self.pixel))
        rescale_col = rescale_img.size[0]
        rescale_row = rescale_img.size[1]
        
        # 2. Imperfect Resize (Aspect Ratio Issue) ---> Zero Padding
        rescale_img = np.array(rescale_img)
        
        # When resizing an image using get_thumbnail(), 
        # if the desired size cannot be achieved due to aspect ratio constraints and the image gradually becomes slightly smaller,
        #  it can be resolved by applying zero padding to maintain the original aspect ratio.
        if rescale_col % self.pixel != 0 :
            pad_pixel = self.pixel - rescale_col % self.pixel 
            rescale_img = np.pad(rescale_img, ((0,0),(0,pad_pixel),(0,0)), 'constant', constant_values=0)
        if rescale_row % self.pixel != 0 :
            pad_pixel = self.pixel - rescale_row % self.pixel 
            rescale_img = np.pad(rescale_img, ((0,pad_pixel),(0,0),(0,0)), 'constant', constant_values=0)
        
        
        np_full_tile = cv2.cvtColor(rescale_img, cv2.COLOR_RGB2BGR)
        
        # save small svs images
        gray_np_full_tile = cv2.cvtColor(rescale_img, cv2.COLOR_RGB2GRAY)        
        back = np.zeros(gray_np_full_tile.shape, dtype= gray_np_full_tile.dtype)

        # 3. Generating a Tile Heatmap based on the Stored Locations and Scores of all Tiles
        # save all tile (Heatmap)
        total_tiles = len(slide2pos)
        for i in range(total_tiles):    # T
            col, row = slide2pos[i]['pos'][0], slide2pos[i]['pos'][1]
            color = 255 * slide2pos[i]['score']
            for k in range(self.pixel-1):
                for j in range(self.pixel-1):
                    if self.pixel * row + k >= back.shape[0] or self.pixel * col + j >= back.shape[1]:
                        print(row, col, k, j )
                    back[self.pixel * row + k][self.pixel * col + j] = color
            
        back = cv2.applyColorMap(back, cv2.COLORMAP_JET)
        alpha = 0.3
        blend = cv2.addWeighted(back, alpha, np_full_tile, 1-alpha, 0)
        cv2.imwrite(ospj(self.result_path, str(self.pixel) + '_all', self.slide + '_' + self.label+ '.jpeg'), blend)

        # save all tile(max_min tile highlight, Heatmap)
        back = np.zeros(gray_np_full_tile.shape, dtype= gray_np_full_tile.dtype)
        slide2pos_score = sorted(slide2pos, key = lambda x : x['score'], reverse=True)
        slide2pos_score = slide2pos_score[:100] + slide2pos_score[-100:]
        total_tiles = len(slide2pos_score)
        for i in range(total_tiles):    
            col, row = slide2pos_score[i]['pos'][0], slide2pos_score[i]['pos'][1]
            color = 255 * slide2pos_score[i]['score']
            for k in range(self.pixel-1):
                for j in range(self.pixel-1):
                    back[self.pixel * row + k][self.pixel * col + j] = color
                    
        back = cv2.applyColorMap(back, cv2.COLORMAP_JET)
        alpha = 0.3
        blend = cv2.addWeighted(back, alpha, np_full_tile, 1-alpha, 0)
        cv2.imwrite(ospj(self.result_path, str(self.pixel) + '_all(max_min)', self.slide + '_' + self.label+ '.jpeg'), blend)

    # save max & min tile (20)
    def max_min(self, args, slide2pos_score):
        slide2pos_score = slide2pos_score[:20] + slide2pos_score[-20:]

        # save max tile
        tiles_path = []
        for i, slide in enumerate(slide2pos_score[:20]):
            col, row = str(slide['pos'][0]), str(slide['pos'][1])
            tile_name = self.slide + '_' + col + '_' + row + '.jpeg'
            tile_path = ospj(args.tile_dir, self.slide, tile_name)
            tiles_path.append(tile_path) 
        pil_images = [Image.open(f) for f in tiles_path]
        tensor_images = [transforms.ToTensor()(img) for img in pil_images]
        grid = torch.stack(tensor_images)
        grid = make_grid(grid, nrow=10)
        save_image(grid, ospj(self.result_path, str(args.pixel) +  '_max_tile', self.slide + '_max_20.png'))

        # save min tile
        tiles_path = []
        for i, slide in enumerate(slide2pos_score[-20:]):
            col, row = str(slide['pos'][0]), str(slide['pos'][1])
            tile_name = self.slide + '_' + col + '_' + row + '.jpeg'
            tile_path = ospj(args.tile_dir, self.slide, tile_name)
            tiles_path.append(tile_path)
        pil_images = [Image.open(f) for f in tiles_path]
        tensor_images = [transforms.ToTensor()(img) for img in pil_images]
        grid = torch.stack(tensor_images)
        grid = make_grid(grid, nrow=10)
        save_image(grid, ospj(self.result_path, str(args.pixel) +  '_min_tile', self.slide + '_min_20.png'))




    
        