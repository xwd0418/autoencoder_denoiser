"""
this file is used to convert the real T1 noise to some arrays and save them
saved as original shape, black and white
    
"""

import json
import os
from model_factory import get_model
from weakref import ref
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import shutil
import matplotlib.image
from glob import glob

orig_img_dir = "/home/wangdong/autoencoder_denoiser/dataset/real_noise"
new_img_dir = orig_img_dir+"_binary_array"
os.makedirs(new_img_dir, exist_ok = True) 
for img_folder in glob(orig_img_dir+"/*/"):
    new_sub_dir = new_img_dir+"/"+img_folder.split("/")[-2]
    os.makedirs(new_sub_dir, exist_ok=True)
    for img_path in glob(img_folder+"*"):
        img = Image.open(img_path)
        gray = img.convert('L')
        binary = gray.point(lambda x: 0 if x>200 else 255, '1')
        np_img = np.array(binary).astype(np.int8)
          
        saved_path = img_path.replace(".png",".npy").replace(".jpg",".npy").replace("real_noise","real_noise_binary_array")
        np.save(saved_path, np_img)
