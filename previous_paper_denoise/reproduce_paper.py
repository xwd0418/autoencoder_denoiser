# make dataset
import os,sys
from glob import glob
import cv2
import torch, copy
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

from tqdm import tqdm
import sys
sys.path.append('../autoencoder_denoiser')
from model_factory import *


class One_D_Dataset(Dataset):
    def __init__(self, split="training", config=None):
        
        self.noise = [ os.path.join("/root/autoencoder_denoiser/previous_paper_denoise/gen_dataset/1d", split, f)\
            for f in os.listdir(os.path.join("/root/autoencoder_denoiser/previous_paper_denoise/gen_dataset", "1d", split))]

        split+="_good"
        self.ref = [ os.path.join("/root/autoencoder_denoiser/previous_paper_denoise/gen_dataset/1d", split, f)\
            for f in os.listdir(os.path.join("/root/autoencoder_denoiser/previous_paper_denoise/gen_dataset", "1d", split))]
        

    def __len__(self):
        return len(self.noise)

    def __getitem__(self, i):
        file_index = i//4
        
        y_noise = np.loadtxt(self.noise[file_index])[i-4*file_index]
        y_ref = np.loadtxt(self.ref[file_index])
        return y_ref, y_noise
    


# get model

specs = ( [64, 128, 1024],
        # 1024, 1024, 1024, 512,
        # 512, 512, 512, 512,
        # ], 
        #  [512, 512, 512, 512,
        # 512, 1024, 1024, 1024,
        [ 512, 256, 64])
    
# model = UNet_Single(1,1,False,1,channel_specs= None)
model = UNet(1,1,False, oneD=True)
model = torch.nn.DataParallel(model).to("cuda")

# config
saved_dir = "/root/autoencoder_denoiser/previous_paper_denoise/oneD_sample_orig_UNet"

curr_epoch = 0

epoch = 200
lr = 0.00001
lr_step = 15
lr_gamma = 0.01
betas = (0.7, 0.9)
batch=4

optimizer = torch.optim.Adam(model.parameters(),betas=betas ,lr = lr) 
# optimizer = torch.optim.SGD(model.parameters(),momentum=0 ,lr = lr, weight_decay=0.01) 
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_step, gamma=lr_gamma)
criterion = torch.nn.MSELoss()

shuffle=False
train_loader = DataLoader(One_D_Dataset("training" ), batch_size=batch, shuffle=shuffle, num_workers=os.cpu_count())
val_loader = DataLoader(One_D_Dataset("validation"), batch_size=batch, shuffle=shuffle, num_workers=os.cpu_count())
test_loader = DataLoader(One_D_Dataset("testing"), batch_size=batch, shuffle=shuffle, num_workers=os.cpu_count())

train_losses = 0
val_losses = 0
test_losses = 0
train_snr = 0
val_snr = 0
test_snr  = 0

# SNR helper
def compute_SNR( raw, noisy_img):
        
        signal = torch.mean(torch.abs(raw))
        noise =  torch.mean((noisy_img - raw)**2)
        noise = torch.sqrt(noise)
        return (signal/noise).item()
    
def SNR_increase(raw, noise, prediction):
        orig_SNR = compute_SNR(raw, noise)
        denoised_SNR = compute_SNR(raw, prediction)
        return denoised_SNR/orig_SNR
    



def one_step(mode='train', show_idx = 0):
        if mode=='train':
            model.train()
            loader = train_loader
            losses = train_losses
            snr = train_snr
        elif mode=='val':
            model.eval()
            loader = val_loader
            losses = val_losses
            snr = val_snr    
        elif mode == 'test':
            model.eval()
            loader = test_loader
            losses = test_losses
            snr = test_snr
            
        losses = 0
        snr = 0
        # temp
        # Iterate over the data, implement the training function
        for iter, data in enumerate((loader)):
            raw, noise = data
            raw, noise = raw.cuda().float(), noise.cuda().float()
            
            noise = torch.unsqueeze( noise,1)
            raw = torch.unsqueeze(raw, 1)
            # print("noise.shape: ",noise.shape)
            prediction = model.forward(noise)
            # print("predcition.shape: ",noise.shape)
            loss = criterion(prediction,raw )
            
            
                    
            if mode == 'train': 
                loss.backward()
                optimizer.step()
            elif iter == show_idx:
                plt.clf()
                ax = plt.subplot(3, 1, 1)
                # plt.tight_layout()
                ax.set_title('orig')
                # ax.axis('off')
                plt.plot(raw[0,0].cpu())

                ax = plt.subplot(3, 1, 2)
                # plt.tight_layout()
                ax.set_title('noise')
                # ax.axis('off')
                plt.plot(noise[0,0].cpu())
                
                ax = plt.subplot(3, 1, 3)
                # plt.tight_layout()
                ax.set_title('predicted')
                # ax.axis('off')
                plt.plot(prediction[0,0].detach().cpu())

                plt.savefig(os.path.join(saved_dir, f"epoch_{curr_epoch}_val_{SNR_increase( raw[0], noise[0], prediction[0])}.png"))


            losses += loss.item()
            snr += SNR_increase( raw, noise, prediction)
        losses /= (iter+1)
        snr /= (iter+1)
        
        return losses ,snr
    
    
train_loss_list = []
train_snr_list = []
val_loss_list = []
val_snr_list = []

import os
os.makedirs(saved_dir, exist_ok=True)

for i in tqdm(range(epoch)):  # loop over the dataset multiple times
            curr_epoch+=1
            print("epoch: ", curr_epoch)
            train_loss, train_accu = one_step("train")
            val_loss,val_SNR_increase = one_step("val", show_idx=np.random.randint(0,5))
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            train_snr_list.append(train_accu)
            val_snr_list.append(val_SNR_increase)
            print("val snr increase is: ", round(val_SNR_increase,2), "val loss is ", round(val_loss,2),  "train loss is ", round(train_loss,2))
            
            lr_scheduler.step()