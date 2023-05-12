import os
from os.path import join as ospj
import staintools
import glob
from PIL import Image
import staintools
import tqdm
from multiprocessing import Pool

METHOD = 'vahadane'
STANDARDIZE_BRIGHTNESS = True
TARGET_IMG = '/data1/wsi/prepared_dataset/total/S20-06726/S20-06726_200_115.jpeg'
TCGA_PATH = '/home/dhlee/Chowder_Practice/0126_tcga'
OUTPUT_PATH = '/home/dhlee/Chowder_Practice/test_tiles_norm/' + METHOD

os.makedirs(OUTPUT_PATH, exist_ok=True)

if __name__ == '__main__':
    i1 = staintools.read_image(TARGET_IMG)
    if STANDARDIZE_BRIGHTNESS:
        i1 = staintools.LuminosityStandardizer.standardize(i1)
    normalizer = staintools.StainNormalizer(method=METHOD)
    normalizer.fit(i1)
    
    TCGA_dirs = os.listdir(TCGA_PATH)
    
    for TCGA_dir in tqdm.tqdm(TCGA_dirs):
        TCGA_dir_path = ospj(TCGA_PATH, TCGA_dir)
        TCGA_dir_output_path = ospj(OUTPUT_PATH, TCGA_dir)
        os.makedirs(TCGA_dir_output_path, exist_ok=True)
    
        TCGA_files = os.listdir(TCGA_dir_path)
        print(f'TCGA_dir: {TCGA_dir}')
        for TCGA_file in tqdm.tqdm(TCGA_files, leave=False, total=len(TCGA_files)):
            TCGA_file_path = ospj(TCGA_dir_path, TCGA_file)
            TCGA_file_output_path = ospj(TCGA_dir_output_path, TCGA_file)
            i2 = staintools.read_image(TCGA_file_path)
            if STANDARDIZE_BRIGHTNESS:
                i2 = staintools.LuminosityStandardizer.standardize(i2)
            i2 = normalizer.transform(i2)
            i2 = Image.fromarray(i2)
            i2.save(TCGA_file_output_path)
        
    
