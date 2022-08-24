import os,sys
from glob import glob
import cv2
import torch, copy
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from data_preprocess import triangle_tessellate , expand


class HSQCDataset(Dataset):
    def __init__(self, split="train", config=None):
        self.dir = "/root/data/hyun_fp_data/hsqc_ms_pairs/"
        self.split = split
        self.config = config
        self.augment = config['dataset'].get("data_augment")
        self.augment = 1 if self.augment==None else int(self.augment)
        assert (self.augment >= 1)
        # self.orig_hsqc = os.path.join(self.dir, "data")
        # assert(os.path.exists(self.orig_hsqc))
        assert(split in ["train", "val", "test"])
        
        self.hsqc_path = 'HSQC_plain_imgs_toghter'
        if self.config['dataset'].get("large_input"):
            self.hsqc_path = 'HSQC_plain_imgs_toghter_1800_1200'
            
        self.hsqc_files = list(os.listdir(os.path.join(self.dir, split, self.hsqc_path)))
        # self.HSQC_files = list(os.listdir(os.path.join(self.dir, split, "HSQC")))
        # assert (len(self.FP_files ) == (self.HSQC_files))

    def __len__(self):
        return len(self.hsqc_files)*self.augment

    def __getitem__(self, i):
        file_index = i//self.augment

        raw_sample = torch.load(os.path.join(self.dir,  self.split, self.hsqc_path, self.hsqc_files[file_index]))
        raw_sample = np.array(raw_sample, dtype="float32")
        upscale_factor = self.config['dataset'].get('signal_upscale')
        if upscale_factor!=None:
            raw_sample = cv2.resize(raw_sample[0], (raw_sample.shape[2]*2,raw_sample.shape[1]*2), interpolation = cv2.INTER_AREA) 
        if self.config['dataset'].get('noise_factor') != None:
            if self.config["dataset"]["noise_factor"] == "random":
                noise_factor = random.uniform(0.2,0.6)
            else:
                noise_factor = self.config["dataset"]["noise_factor"]
        else :
            noise_factor = 1


        # add noise based on provided noise type
        if self.config["dataset"]["noise_type"] == "random":
            self.config["dataset"]["noise_type"] = random.choice(["gaussian", "white" ])
        elif self.config["dataset"]["noise_type"] == "gaussian":
            noisy_sample = raw_sample + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=raw_sample.shape)
        elif self.config["dataset"]["noise_type"] == "white":    
            noisy_sample = raw_sample + noise_factor * np.random.uniform(low=0.0, high=1.0, size=raw_sample.shape)
        elif self.config["dataset"]["noise_type"] == "t1": 
            noisy_sample = add_t1_noise(raw_sample, self.config)
            white_noise_rate=self.config['dataset'].get('white_noise_rate')
            if white_noise_rate is not None:
                noisy_sample += self.config["dataset"]["noise_factor"] * np.random.binomial(1, white_noise_rate,  size=raw_sample.shape)


        else:
            raise Exception("unkown type of noise {}".format(self.config["dataset"]["noise_type"]))
       
        noisy_sample = np.clip(noisy_sample, 0., 1.)


        if self.config['model']['model_type'] != 'filter' and self.config['model']['model_type'] != 'vanilla':
            raw_sample = np.expand_dims(raw_sample, axis=0)
            noisy_sample = np.expand_dims(noisy_sample, axis=0)

        if self.config["dataset"]["pre-filtered"]:
            noisy_sample = np.array([[[filtering(float(k)) for k in j] for j in i] for i in noisy_sample])
        
        if self.config["dataset"]["tessellate"]:
            
            selected_noisy_sample = np.array([[[selecting(float(k)) for k in j] for j in i] for i in noisy_sample])
            tessellated_noise = triangle_tessellate( selected_noisy_sample[0], self.config["dataset"]["upscale"])
            tessellated_noise = np.expand_dims(tessellated_noise, axis=0)
            # tessellated_noise = np.expand_dims(tessellated_noise, axis=0)
            # noisy_sample = np.stack((expand(noisy_sample), tessellated_noise), axis=0)
            
            ### using low resolution of tessellation 
            if self.config['model']['model_type'] == "UNet_2": 
                concat = np.concatenate((noisy_sample, tessellated_noise))                
                return raw_sample, concat
            ### using Jnet 
            else:
                return raw_sample, (noisy_sample, tessellated_noise)
        
        return raw_sample,noisy_sample

class RealNoiseDataset(Dataset):
    def __init__(self, config):
        self.imgs = []
        orig_img_dir = "/root/autoencoder_denoiser/dataset/real_noise"
        # new_img_dir = orig_img_dir+"_binary_array"
        new_img_dir = orig_img_dir+"_greyscale"

        img_shape = (120 *config['dataset']['signal_upscale'] ,180 * config['dataset']['signal_upscale']) 

        for img_folder in glob(new_img_dir+"/*/"):
            for img_path in glob(img_folder+"/*"):
                img = np.load(img_path)
                img = cv2.resize( np.array(img), img_shape, interpolation = cv2.INTER_AREA) 
                self.imgs.append(img)
        
    def  __len__(self):
        return len(self.imgs)
        
    def __getitem__(self, index):
        return self.imgs[index]

def get_datasets(config):
    shuffle=config["dataset"]['shuffle']
    batch = config["dataset"]['batch_size']
    train_loader = DataLoader(HSQCDataset("train", config), batch_size=batch, shuffle=shuffle, num_workers=os.cpu_count())
    val_loader = DataLoader(HSQCDataset("val",config), batch_size=batch, shuffle=shuffle, num_workers=os.cpu_count())
    test_loader = DataLoader(HSQCDataset("test",config), batch_size=batch, shuffle=shuffle, num_workers=os.cpu_count())
    return train_loader, val_loader , test_loader

def get_real_img_dataset(config):
    batch = config["dataset"]['batch_size']
    shuffle=config["dataset"]['shuffle']
    return DataLoader(RealNoiseDataset(config), batch_size=batch, shuffle=shuffle, num_workers=os.cpu_count())




"""helper functioner to gerneate noise"""

def selecting(x):
    if x>=0.6: return 1
    return 0

def filtering(x):
    if x>=0.6: return x
    return 0


def add_t1_noise(img, config):
    height = img.shape[0]
    streak_p = config['dataset'].get('streak_prob')  # probability of generating streak noise
    if config['dataset'].get('noise_factor') != None:
        if config["dataset"]["noise_factor"] == "random":
            noise_factor = random.uniform(0.05,0.1)
        else:
            noise_factor = config["dataset"]["noise_factor"]
    else :
        noise_factor = 1
    
    noisy_img = copy.deepcopy(img)
    cross_noise, cross_points = generate_cross_noise(img, config)
    noisy_columns = cross_points[1]
    for col in noisy_columns:
        if np.random.binomial(1, streak_p):
            noise_rate = np.random.binomial(height, np.random.uniform(low=0.2, high=0.7))/ height
            noise =  np.random.binomial(1, noise_rate, height)
            noisy_img[:,col] += noise*noise_factor
            
    noisy_img = noisy_img + cross_noise*noise_factor
        
    return noisy_img
        

# def add_noise_to_column(col, cross_points, height, streak_p, noise_factor):
#     for _ in range(int(np.sum(col))):
#         if np.random.binomial(1, streak_p):
#             noise_rate = np.random.binomial(height, np.random.uniform(low=0.1, high=0.7))/ height
#             noise =  np.random.binomial(1, noise_rate, height) * (random.uniform(0.2,0.6)  if noise_factor=="random" else noise_factor)
#             col = col+noise
        
#     return col

def generate_cross_noise(img, config):
    noise_probability = config['dataset']['cross_prob']
    output_noise = np.zeros(img.shape)
    points =  np.array(np.where(img==1))
    points = points[:, np.random.permutation(points.shape[1])]
    points = points[:, 0:int(len(points[0])*noise_probability)]
    if len(points) == 0: 
        return output_noise
    
    
    cross_length = np.clip(np.random.poisson(3), 1, 12)
    # adding horizontal cross
    for index_shift  in range(-1*cross_length, cross_length+1):
        if index_shift == 0 :
            continue
        points_copy1 = copy.deepcopy(points)
        points_copy1[1] += index_shift
        points_copy1[1] = np.clip(points_copy1[1], 0, img.shape[1]-1)
        noise_layer1 = np.zeros(img.shape)
        noise_layer1[tuple(points_copy1)] = 1
        output_noise += noise_layer1
    
    # somtimes crosses are elongated at one direction
    if np.random.binomial(1, 0.3):
        cross_length = np.clip(np.random.poisson(3), 1, 12)        
    # adding vertical
    for index_shift  in range(-1*cross_length, cross_length+1): 
        if index_shift == 0 :
            continue
        points_copy2 = copy.deepcopy(points)
        points_copy2[0] += index_shift
        points_copy2[0] = np.clip(points_copy2[0], 0, img.shape[1]-1)
        noise_layer2 = np.zeros(img.shape)
        noise_layer2[tuple(points_copy2)] = 1
        output_noise += noise_layer2
        
    return output_noise, points