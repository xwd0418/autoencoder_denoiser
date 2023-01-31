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
from experiment import Metric, compute_metrics
from torch.utils.tensorboard import SummaryWriter


class One_D_Dataset(Dataset):
    def __init__(self, split="training", config=None):
        
        self.noise = [ os.path.join("/root/autoencoder_denoiser/previous_paper_denoise/gen_dataset/1d", split, f)\
            for f in os.listdir(os.path.join("/root/autoencoder_denoiser/previous_paper_denoise/gen_dataset", "1d", split))]

        split+="_good"
        self.ref = [ os.path.join("/root/autoencoder_denoiser/previous_paper_denoise/gen_dataset/1d", split, f)\
            for f in os.listdir(os.path.join("/root/autoencoder_denoiser/previous_paper_denoise/gen_dataset", "1d", split))]
        
        self.ref = sorted(self.ref)
        self.noise = sorted(self.noise)

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
    
model = UNet_Single(1,1,False,1,channel_specs= None)
# model = UNet(1,1,False, oneD=True)
model = torch.nn.DataParallel(model).to("cuda")

# config
saved_dir = "/root/autoencoder_denoiser/previous_paper_denoise/oneD_lowerLR"
os.makedirs(saved_dir, exist_ok=True)
log_file = open(saved_dir+"/log.txt", "w")

curr_epoch = 0

epoch = 200
lr = 0.00001
lr_step = 5
lr_gamma = 0.01
betas = (0.7, 0.999)
batch=4

optimizer = torch.optim.Adam(model.parameters(), betas= betas ,lr = lr) 
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

train_metric = Metric()
val_metric = Metric()
test_metric = Metric()


writer = SummaryWriter(log_dir=saved_dir)
curr_iter = 0


def one_step(mode='train', show_idx = 0):
        if mode=='train':
            model.train()
            loader = train_loader
            losses = train_losses
            metric = train_metric
        elif mode=='val':
            model.eval()
            loader = val_loader
            losses = val_losses
            metric = val_metric  
        elif mode == 'test':
            model.eval()
            loader = test_loader
            losses = test_losses
            metric = test_metric
            
        losses = 0
        # temp
        # Iterate over the data, implement the training function
        for iter, data in enumerate(tqdm(loader)):
            metric.reset()
            global curr_iter
            curr_iter+=1
            optimizer.zero_grad()
            raw, noise = data
            raw, noise = raw.cuda().float(), noise.cuda().float()
            
            # noise = torch.unsqueeze( noise,1)
            # raw = torch.unsqueeze(raw, 1)
            # print("noise.shape: ",noise.shape)
            prediction = model.forward(noise).double()
            # print(prediction)
            prediction = torch.where(prediction>0.0, prediction, 0.0)
            prediction /= torch.unsqueeze(torch.max(prediction, 1)[0], 1)
            prediction = prediction.float()
            # print("predcition.shape: ",noise.shape)
            loss = criterion(prediction,raw )
            
            
                    
            if mode == 'train': 
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                if iter == show_idx:
                    plt.clf()
                    ax = plt.subplot(3, 1, 1)
                    # plt.tight_layout()
                    ax.set_title('orig')
                    # ax.axis('off')
                    plt.plot(raw[0].cpu())

                    ax = plt.subplot(3, 1, 2)
                    # plt.tight_layout()
                    ax.set_title('noise')
                    # ax.axis('off')
                    plt.plot(noise[0].cpu())
                    
                    ax = plt.subplot(3, 1, 3)
                    # plt.tight_layout()
                    ax.set_title('predicted')
                    # ax.axis('off')
                    plt.plot(prediction[0].detach().cpu())

                    plt.savefig(os.path.join(saved_dir, f"epoch_{curr_epoch}_{mode}.png"))

            
                for i in range(len(raw)):
                    metric.update(raw[i].detach(), noise[i].detach(), prediction[i].detach())
                    # a = torch.rand(10)
                    # metric.update(a,a,a)
                  
            writer.add_scalar(f'{mode}/loss', loss, curr_iter)        
            losses += loss.item()
            metric.avg(batch)
            metric.write(writer, mode, curr_iter)
            
        losses /= (iter+1)
        
        
        # msg = f"{mode} SNR_orig is:  {round(metric.snr_orig,3)}, \
        #         {mode} SNR_denoised is:  {round(metric.snr_denoised,3)}, \
        #         {mode} SNR_increase is:  {round(metric.snr_inc,3)}, \
        #         {mode} wSNR_orig is:  {round(metric.wsnr_orig,3)}, \
        #         {mode} wSNR_denoised is:  {round(metric.wsnr_denoised,3)}, \
        #         {mode} wSNR_increase is:  {round(metric.wsnr_inc,3)}, \
        #         {mode} loss is  {round(losses,5)} \n"


        
        return losses , None
    
    
early_stop = 0
best_val_loss = 99999
for i in (range(epoch)):  # loop over the dataset multiple times
            curr_epoch+=1
            print("epoch: ", curr_epoch)
            
            train_loss, train_msg = one_step("train")
            with torch.no_grad():
                val_loss, val_msg = one_step("val", show_idx=1)
                if best_val_loss >= val_loss:
                    best_val_loss = val_loss
                    early_stop = 0
                else:
                    early_stop+=1
                    if early_stop==5:
                        print("early stoppedZZz")
                        break

            
            # total_msg = train_msg+val_msg
            # print(total_msg)
            # log_file.write(total_msg)
            
            lr_scheduler.step()