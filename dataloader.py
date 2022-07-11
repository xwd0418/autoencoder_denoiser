import os,sys
import torch, copy
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from tessellate import triangle_tessellate , expand


# imgs = []
print("loading data ...")
# for i in range(1,20):
#     data = np.load("dataset/Jeol_info{}000.npy".format(str(i)),allow_pickle=True)
#     img_data = data[:,3]
#     imgs.append(img_data)
#     # the second index has to be 3 to show be some image
all_data  = np.load('dataset/single.npy',allow_pickle=True)

print("finish loading")

class HSQC_Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, all_data, config=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.all_data = all_data
        self.config = config

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        raw_sample = self.all_data[idx]
        if self.config:
            if self.config["dataset"]["noise_factor"] == "random":
                noise_factor = random.uniform(0.2,0.6)
            else:
                noise_factor = self.config["dataset"]["noise_factor"]
        else :
            noise_factor = 0.3


        # add noise based on provided noise type
        if self.config["dataset"]["noise_type"] == "random":
            self.config["dataset"]["noise_type"] = random.choice(["gaussian", "white" ])
        elif self.config["dataset"]["noise_type"] == "gaussian":
            noisy_sample = raw_sample + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=raw_sample.shape)
        elif self.config["dataset"]["noise_type"] == "white":    
            noisy_sample = raw_sample + noise_factor * np.random.uniform(low=0.0, high=1.5, size=raw_sample.shape)
        elif self.config["dataset"]["noise_type"] == "t1": 
            noisy_sample = add_t1_noise(raw_sample, self.config)
            noisy_sample += self.config["dataset"]["white_noise_factor"] * np.random.uniform(low=0.0, high=1.5, size=raw_sample.shape)
       
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

 

def get_datasets(config):
    # np.random.shuffle(all_data)
    first = int(len(all_data)*0.8)
    second = int(len(all_data)*0.9)

    train = all_data[:first]
    val = all_data[first:second]
    test = all_data[second:]
    
    augment = config["dataset"].get('data_augment')
    
    if augment == None:
        print("Not using data augmentation")
    else:
        train, val, test = np.tile(train,5), np.tile(val,5), np.tile(test,5)
    


    shuffle=config["dataset"]['shuffle']
    batch = config["dataset"]['batch_size']
    train_loader = DataLoader(HSQC_Dataset(train,config), batch_size=batch, shuffle=shuffle, num_workers=os.cpu_count())
    val_loader = DataLoader(HSQC_Dataset(val,config), batch_size=batch, shuffle=shuffle, num_workers=os.cpu_count())
    test_loader = DataLoader(HSQC_Dataset(test,config), batch_size=batch, shuffle=shuffle, num_workers=os.cpu_count())

    return train_loader, val_loader , test_loader


"""some helper functions to pre_process data"""
def selecting(x):
    if x>=0.6: return 1
    return 0

def filtering(x):
    if x>=0.6: return x
    return 0

def add_t1_noise(img, config):
    height = img.shape[0]
    streak_p = config['dataset'].get('streak_prob')  # probability of generating streak noise
    noise_factor = config['dataset']['noise_factor']
    
    noisy_img = np.apply_along_axis(add_noise_to_column, 0, img.astype(float), height, streak_p, noise_factor)
    
    if config['dataset'].get("cross_prob") != None:
        cross_noise = generate_cross_noise(img, config)
        noisy_img += cross_noise
    return noisy_img
        

def add_noise_to_column(col, height, streak_p, noise_factor):
    for _ in range(int(np.sum(col))):
        if np.random.binomial(1, streak_p):
            noise_rate = np.random.binomial(height, np.random.uniform(low=0.1, high=0.7))/ height
            noise =  np.random.binomial(1, noise_rate, height) * (random.uniform(0.2,0.6)  if noise_factor=="random" else noise_factor)
            col = col+noise
        
    return col

def generate_cross_noise(img, config):
    noise_probability = config['dataset']['cross_prob']
    output_noise = noise_layer = np.zeros(img.shape)
    points =  np.array(np.where(img==1))
    points = points[:, 0:int(len(points[0])*noise_probability)]
    if len(points) == 0: 
        return output_noise
    for noise_rate, index_shift  in [(0.1,-3), 
                                     (0.2,-2), 
                                     (0.3,-1),
                                     (0.3,1),
                                     (0.2,2),
                                     (0.1,3)]:
        points_copy1 = copy.deepcopy(points)
        points_copy1[1] += index_shift
        points_copy1[1] = np.clip(points_copy1[1], 0, img.shape[1]-1)
        noise_layer1 = np.zeros(img.shape)
        noise_layer1[tuple(points_copy1)] = noise_rate
        output_noise += noise_layer1
        
        points_copy2 = copy.deepcopy(points)
        points_copy2[0] += index_shift
        points_copy2[0] = np.clip(points_copy2[0], 0, img.shape[1]-1)
        noise_layer2 = np.zeros(img.shape)
        noise_layer2[tuple(points_copy2)] = noise_rate
        output_noise += noise_layer2
        
    return output_noise