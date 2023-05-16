import os
from os.path import join as ospj
import staintools
import glob
from PIL import Image
import staintools
import tqdm
from multiprocessing import Pool
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', type=str, default='../Data/Tile')
    parser.add_argument('-d', '--destination', type=str, default='../Data/Norm_tile')
    args = parser.parse_args()
    
    i1 = staintools.read_image("target.jpeg")
    i1 = staintools.LuminosityStandardizer.standardize(i1)
    normalizer = staintools.StainNormalizer(method='vahadane')
    normalizer.fit(i1)
    
    source_dirs = os.listdir(args.source)
    
    for source in tqdm.tqdm(source_dirs):
        source_dir_path = ospj(args.source, source)
        destination_path = ospj(args.destination, source)
        os.makedirs(destination_path, exist_ok=True)
    
        source_files = os.listdir(source_dir_path)
        print(f'source_dir: {source}')
        for source_file in tqdm.tqdm(source_files, leave=False, total=len(source_files)):
            source_file_path = ospj(source_dir_path, source_file)
            norm_img_path = ospj(destination_path, source_file)
            i2 = staintools.read_image(source_file_path)
            i2 = staintools.LuminosityStandardizer.standardize(i2)
            i2 = normalizer.transform(i2)
            i2 = Image.fromarray(i2)
            i2.save(norm_img_path)



        
    
