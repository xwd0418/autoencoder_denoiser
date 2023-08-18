"""Convert png files to numpy arrays, which is used to build dataset"""


import os, shutil
img_dir = "/root/autoencoder_denoiser/dataset/real_img_referral_for_testing"
clean_folder_name = "bitmap_real_hsqc_clean"
clean_dir = os.path.join(img_dir, clean_folder_name)
noise_foler_name = "super_noisy_1"
noisy_dir = os.path.join(img_dir, noise_foler_name)

save_dir_resize = f'/root/autoencoder_denoiser/dataset/resized_super_noisy_1/'
save_dir_orig_size = f'/root/autoencoder_denoiser/dataset/orig_size_super_noisy_1/'

import os,sys
from glob import glob
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

import matplotlib.image
clist = [(0,"darkblue"), (0.5,"white"), (1, "darkred")]
custom_HSQC_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("_",clist)

# plt.rcParams["figure.figsize"] = (40,90)

imgs_resized=[]
imgs_orig_size=[]
all_img_paths = (glob(noisy_dir+"/*"))
os.makedirs(save_dir_resize, exist_ok=True)
os.makedirs(save_dir_orig_size, exist_ok=True)

last_compound_name = ''
iter = 0

def convert_to_intensities(custom_HSQC_cmap, path):
    print(path)
    img = cv2.imread(path)
    img_truth = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("float32")
    # print("ground truth shape",img_truth.shape)
    # img_truth = cv2.resize(img_truth.astype("float32"), (240, 360))
    R, G, B = img_truth[:,:,0], img_truth[:,:,1], img_truth[:,:,2]
    shape = img_truth.shape[0:2]
    plus = R
    minus = B
            # plus = np.zeros(shape)
            # minus = np.zeros(shape)
            # plus_pos = B <= 0#np.logical_and( R-B>100 , B <100 ) 
            # plus[plus_pos]=R[plus_pos]
            # minus_pos = R <= 0 # np.logical_and( B-R>100 , R <100 ) 
            # minus[minus_pos] = B[minus_pos]
    
    plus_groud = plus/255 # I will assume this is plus but not sure
    minus_groud = minus/255
    ground_truth = plus_groud - minus_groud
    # norm_image = cv2.normalize(image, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    # figure, plots = plt.subplots(ncols=2, nrows=1)
    # plots[0].imshow(ground_truth_resized.astype(int))

    # plots[1].imshow(ground_truth, cmap=custom_HSQC_cmap, vmax=1, vmin=-1)
    return ground_truth

for img_path in tqdm(all_img_paths):
    # print(img_path)
    iter+=1
    noise_level = img_path.split('/')[-1].split('_')[-1]
    noise_level = noise_level[:-4]
    compound_name = img_path.split('/')[-1].split('_')[0]
   

    '''ground truth'''
    ground_path = img_path.replace(noise_foler_name,clean_folder_name)\
                    .replace("_ noisy_", "_noisy_").replace("noisy_","original_")

    ground_path = ground_path[:-5] + '1.png'
    if ground_path[-6].isdigit():
        # print(ground_path)
        ground_path = ground_path[:-6]+ground_path[-5:]
    # print(gound_path)
    # print(ground_path)
    ground_truth = convert_to_intensities(custom_HSQC_cmap, ground_path)
    resized_truth = cv2.resize(ground_truth, (240, 360))
   
    """noise"""
    noise_input = convert_to_intensities(custom_HSQC_cmap, img_path)
    resized_noise = cv2.resize(noise_input, (240, 360))
                            
    noise_level = img_path.split('/')[-1].split('_')[-1]
                                # # print("noise shape",img.shape)
                                # print(noise_level)
                                
                                
    name = img_path.split("/")[-1].split(".")[0]
    name = [ord(c) for c in name]

    np.savez_compressed(f'{save_dir_resize}/{compound_name}_{noise_level}.npz', noise=resized_noise, ground_truth=resized_truth, name = name )
    np.savez_compressed(f'{save_dir_orig_size}/{compound_name}_{noise_level}.npz', noise=noise_input, ground_truth=ground_truth, name = name)
    

    # # imgs_resized.append((resized_input,ground_truth_resized ))
    # # imgs_orig_size.append((img_result, ground_truth))
                           
    # if iter ==10 : break
