from openslide import open_slide
import numpy as np
from tqdm import tqdm
import argparse
import pickle
import cv2
import shutil
import os
import pandas as pd
import slideio 

parser = argparse.ArgumentParser()
parser.add_argument('--csvpath', default=None, type=str, help='Target csv file path')
parser.add_argument('--slidepath', default=None, type=str, help='Target Whole slide path')
parser.add_argument('--target-mpp', default=0.5, type=float, help='Target MPP')
parser.add_argument('--outpath', default=None, type=str, help='Save Outpath')
parser.add_argument('--downsample', default=1, 
                    type=int, help='Whole Slide Image donwsample Size')
parser.add_argument('--ratio', default=1, 
                    type=int, help='Crop Image donwsample Size')
parser.add_argument('--stride', default=256, 
                    type=int, help='Crop Image Size')

def load_WSI(path, target_mpp, downsample):
    slide_io = slideio.open_slide(path,'SVS')
    scene_io = slide_io.get_scene(0)
    w, h = scene_io.rect[2:]
    mpp = scene_io.resolution[0] * 1000000
    mpp_ratio = target_mpp / mpp
    adjusted_w = w / mpp_ratio
    adjusted_w = int(adjusted_w// downsample)
    return scene_io, mpp_ratio, adjusted_w, w, h

def return_size(pixel, mpp_ratio, size):
    x = int(pixel[0] * size * mpp_ratio)
    y = int(pixel[1] * size * mpp_ratio)
    w = int(pixel[2] * size * mpp_ratio)
    h = int(pixel[3] * size * mpp_ratio)
    return (x, y, w, h)

def main(csv_name, slidepath, outpath, target_mpp, downsample, ratio, stride, num):
    os.makedirs(os.path.join(outpath), exist_ok=True)
    for idx, data_path in enumerate(tqdm(csv_name)):
        print(f'Start : {data_path}')
        if idx>=num:
            try:
                os.makedirs(os.path.join(outpath, data_path), exist_ok=True)
                scene_io, _, _, width, height = load_WSI(path=os.path.join(slidepath, data_path+'.svs'), target_mpp=target_mpp, 
                                                               downsample=downsample)
                for widx in range(0, width, stride):
                    for hidx in range(0, height, stride):
                        tmp_wstride = stride if widx+stride <= width else width-widx
                        tmp_hstride = stride if hidx+stride <= height else height-hidx
                        if tmp_wstride>stride*0.8 and tmp_hstride>stride*0.8:
                            slide_region = scene_io.read_block((widx, hidx, tmp_wstride, tmp_hstride), size=(int(tmp_wstride//ratio), int(tmp_hstride//ratio)))
                            cv2.imwrite(os.path.join(outpath, data_path, f'{str(widx)}_{str(hidx)}.png'), slide_region)                
            except:
                continue
        
if __name__ == '__main__':
    args = parser.parse_args()
    csv_data = pd.read_csv(args.csvpath)
    csv_name = csv_data['ID'].values
    main(csv_name=csv_name, slidepath=args.slidepath, outpath=args.outpath, target_mpp=args.target_mpp, 
        downsample=args.downsample, ratio=args.ratio, stride=args.stride, num=0)