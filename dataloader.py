import os,sys
import torch
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
                noise_factor = random.uniform(0.1,0.3)
            else:
                noise_factor = self.config["dataset"]["noise_factor"]
        else :
            noise_factor = 0.3


        # add noise based on provided noise type
        if self.config["dataset"]["noise_type"] == "random":
            self.config["dataset"]["noise_type"] = random.choice(["gaussian", "white" ])
        if self.config["dataset"]["noise_type"] == "gaussian":
            noisy_sample = raw_sample + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=raw_sample.shape)
        elif self.config["dataset"]["noise_type"] == "white":    
            noisy_sample = raw_sample + noise_factor * np.random.uniform(low=0.0, high=1.5, size=raw_sample.shape)
        else:
            raise Exception("unkown type of noise {}".format(self.config["dataset"]["noise_type"]))
        noisy_sample = np.clip(noisy_sample, 0., 1.)

        if self.config['model']['model_type'] != 'filter' and self.config['model']['model_type'] != 'vanilla':
            raw_sample = np.expand_dims(raw_sample, axis=0)
            noisy_sample = np.expand_dims(noisy_sample, axis=0)

        if self.config["dataset"]["pre-filtered"]:
            noisy_sample = np.array([[[filtering(float(k)) for k in j] for j in i] for i in noisy_sample])
        
        if self.config["dataset"]["tessellate"]:
            # tessellated_raw = triangle_tessellate( raw_sample)
            # tessellated_raw = np.expand_dims(tessellated_raw, axis=0)
            # raw_sample = np.stack((expand(raw_sample), tessellated_raw), axis=0)
            
            selected_noisy_sample = np.array([[selecting(float(j)) for j in i] for i in noisy_sample])
            tessellated_noise = triangle_tessellate( selected_noisy_sample)
            # tessellated_noise = np.expand_dims(tessellated_noise, axis=0)
            # noisy_sample = np.stack((expand(noisy_sample), tessellated_noise), axis=0)
            
            return raw_sample,( noisy_sample, tessellated_noise)

            
        
        return raw_sample,noisy_sample

 

def get_datasets(config):
    # np.random.shuffle(all_data)
    first = int(len(all_data)*0.8)
    second = int(len(all_data)*0.9)

    train = all_data[:first]
    val = all_data[first:second]
    test = all_data[second:]

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

