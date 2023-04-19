import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from openslide import open_slide
import pickle
import shutil
import ray

def get_mask_slide_path(path):
    mask_path = []
    slide_path = []
    mask_tmp = os.path.join(path, 'masks')
    image_tmp = os.path.join(path, 'slides')
    mask_lists = os.listdir(mask_tmp)
    for image_list in mask_lists:
        slide_path.append(os.path.join(image_tmp, image_list.split('.')[0]+'.svs'))
        mask_path.append(os.path.join(mask_tmp, image_list.split('.')[0]+'.png'))
    return slide_path, mask_path

def find_region_area_contours(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    tmp = np.where(mask==255, 0, 1)
    tmp = np.uint8(tmp)
    contours, _ = cv2.findContours(tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours

def save_slide_contours(slide_path, mask, contours, num_pixel=[0, 29, 97], image_size=512):
    split_path = slide_path.split('/')[-1].split('.')[0]
    slide = open_slide(slide_path)
    for idx, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        if w > image_size and h > image_size:
            slide_region = slide.read_region((8*x, 8*y), 0, (8*w, 8*h))
            mask_region = mask[y:y+h, x:x+w]
            tmp = np.zeros((mask_region.shape[0], mask_region.shape[1]))
            for idx, pixel in enumerate(num_pixel):
                tmp[mask_region==pixel] = 50*(idx+1)
            if (mask_region>0).any():
                cv2.imwrite(os.path.join(mask_output_path, split_path + f'_{idx}.png'), tmp)
                cv2.imwrite(os.path.join(image_output_path, split_path + f'_{idx}.png'), np.uint8(slide_region))

#@ray.remote 
def find_images_qda(img_paths, mask_paths, output_img, output_mask, img_size, thresh, qda):
    threshold = int((img_size**2)*thresh)
    for img_path, mask_path in zip(img_paths, mask_paths):
        tmp_img = img_paths.split('/')[-1]
        tmp_mask = mask_paths.split('/')[-1]
        img = cv2.imread(img_paths, cv2.IMREAD_COLOR)
        w, h, _ = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        reshaped_patch = np.reshape(img, (-1, 3))
        ret_prob = qda.predict_proba(reshaped_patch)   # return posterior probabilities of classification per class (n_sampels, n_classes)
        ret_prob = np.reshape(ret_prob[:, 0], (w, h))   # return likelihood of pixels assigned to class 0 (background)
        img[np.where(ret_prob > 0.7)] = (0, 0, 0)   # change pixel to background
        img[np.where(ret_prob < 0.7)] = (255, 255, 255)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype(np.uint8)
        if np.count_nonzero(img) > thresh:
            shutil.copy(img_paths, os.path.join(output_img, tmp_img))
            shutil.copy(mask_paths, os.path.join(output_mask, tmp_mask))
            
@ray.remote
def patch_embedding_n_score(image_path, path, stride=1024):
    whole_files = {'image':[], , 'image_path': image_path, 'filename':path}
    slide = open_slide(image_path)
    w, h = slide.level_dimensions[0]
    files = []
    for w_idx in range(0, w, stride):
        for h_idx in range(0, h, stride):
            w_stride = stride if w_idx+stride < w else w-w_idx
            h_stride = stride if h_idx+stride < h else h-h_idx
            files.append([w_idx, h_idx, w_stride, h_stride])
    whole_files['image'] = files
    return whole_files