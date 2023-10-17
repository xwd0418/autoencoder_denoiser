import json
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import json
import cv2
import matplotlib.image
from glob import glob
import torch
from model_factory import get_model
import argparse
from model_factory import UNet
from hsqc_dataset import *
from tqdm import tqdm
from utils import display_pics

"""configurations"""
device = torch.device("cuda:0")
version = "bitmap"
config_path = f"/root/autoencoder_denoiser/configs_{version}"
exp_dir = f'/root/autoencoder_denoiser/exps/results_{version}'

class Test():
    def __init__(self, model_name, 
                    config_path = config_path,
                    exp_dir = exp_dir,
                    threshold=None) -> None:
        self.dilation = False
        self.resize = True
        self.config = None
        name = model_name
        self.name = name

        f = open(f'{config_path}/'+ name + '.json')
        self.config = json.load(f)
        if threshold :
            return
        experiment_path = f"{exp_dir}/{name}/" 
        print('load from: ', os.path.join(experiment_path, 'latest_model.pt'))
        state_dict = torch.load(os.path.join(experiment_path, 'latest_model.pt'))
        model = get_model(self.config)
        # try:
        model.load_state_dict(state_dict['model'])
        model = torch.nn.DataParallel(model)
        # except:
        #     model = torch.nn.DataParallel(model)
        #     model.load_state_dict(state_dict['model'])
            
        model.to(device)
        model.eval()
        self.model = model

        


criterion = torch.nn.MSELoss(reduction="sum")


# dann_test_loader = DataLoader(RealNoiseDataset_Byeol(dann_test.config,range_low=0,range_high=2, show_name=True),
                        #  batch_size=1, shuffle=False, num_workers=8)


def compute_SNR(raw, noisy_img): 
    signal_position= torch.where(raw!=0)
    # noise_position= torch.where(raw==0)
    # prediction_error = torch.sum( torch.abs(raw-noisy_img))
 
    avg_signal = torch.sum( torch.abs(raw))/len(signal_position[0])
    noise_std =  torch.std(noisy_img - raw)
    return (avg_signal/noise_std).item()

def test(*model_tests):
    need_display = True
    for model_test in model_tests:
        displayed=0
        display_num = 0
        loss = 0
        snr = 0
        plt.rcParams["figure.figsize"] = (20,10)
        with torch.no_grad():
            for iter, data in enumerate(tqdm(test_loader)):
                noise, raw = data
                if len(raw.shape)==3:   
                    raw, noise = raw.unsqueeze(1), noise.unsqueeze(1)
                raw, noise = raw.to(device).float(), noise.to(device).float()
                prediction = model_test.model.forward(noise)
                
                # find loss
                # prediction = prediction.type(torch.float32)
                ground_truth = raw
            
                # add adv loss !!!
                prediction = torch.clip(prediction,-1,1)

            # print(denoised_1.shape)
            # print(ground_truth.shape)
                loss += criterion(prediction,ground_truth )
                snr += compute_SNR(raw, prediction)
            
                if need_display and displayed<2:
                    noise_pic , prediction_pic, raw_pic = noise[1],prediction[1], raw[1]
                    
                    # print("?")
                    # plt.clf()

                    # print(os.path.join(self._test_samples_path, f"sample_image{displayed}.png"))
                    
                    display_pics(noise_pic[0].cpu(), prediction_pic[0].cpu(), raw_pic[0].cpu())
                    displayed = displayed+1
            
                    
            loss /= len(test_loader.dataset)  
            snr /=   len(test_loader.dataset)
            print("test loader size:" , len(test_loader.dataset))
            print(f"loss of model: {model_test.name} is {loss}")
            print(f"snr of model: {model_test.name} is {snr}")


    
def test_thresholding(test_loader, threshold_value = 0.4, dir_name_to_save = "thresholding"):
    testing_result_dir = f'/root/autoencoder_denoiser/testing_real_imgs_results/{test_loader.dataset.data_folder_name}_{dir_name_to_save}/'
    os.makedirs(testing_result_dir, exist_ok= True)    
    displayed=0
    loss = 0
    snr = 0
    snr_orig = 0
    with torch.no_grad():
        for iter, data in enumerate(tqdm(test_loader)):
            # if iter==1: break
            noise, raw,name = data
            name = "".join([chr(i) for i in name[0]])
            if len(raw.shape)==3:   
                raw, noise = raw.unsqueeze(1), noise.unsqueeze(1)
            prediction = torch.clone(noise)
            prediction[abs(prediction)<threshold_value]=0
        
        # find loss
        # prediction = prediction.type(torch.float32)
            ground_truth = raw
    
        # add adv loss !!!
            prediction = torch.clip(prediction,-1,1)

    # print(denoised_1.shape)
    # print(ground_truth.shape)
            # print (prediction.size())
            # print(ground_truth.size())
            # if prediction.shape!=ground_truth.shape:
            #     print(ground_truth.shape)
            #     exit()
            loss += criterion(prediction,ground_truth )
            snr += compute_SNR(raw, prediction)
            snr_orig += compute_SNR(raw, noise)
    
        # if need_display and displayed<2:
            if True:
                noise_pic , prediction_pic, raw_pic = noise[0],prediction[0], raw[0]
            
            # print("?")
            # plt.clf()

                save_path = f'{testing_result_dir}/result_{name}.png'
                display_pics(noise_pic[0].cpu(), prediction_pic[0].cpu(), raw_pic[0].cpu(), save_path=save_path)
                
                displayed = displayed+1
            
        loss /= len(test_loader.dataset)  
        snr /=   len(test_loader.dataset)
        snr_orig /=   len(test_loader.dataset)

        print("test loader size:" , len(test_loader.dataset))
        print(f"loss of model: thresholding is {loss}")
        print(f"snr of model: thresholding is {snr}")



match_hist_baseline_test = Test("match_hist")
test_loader = DataLoader(RealNoiseDataset_Byeol(match_hist_baseline_test.config,range_low=0,range_high=100, show_name=True),
                        batch_size=1, shuffle=False, num_workers=8)
if __name__ == "__main__":
    # my_model_test = Test("t1_03")
    # paper_1d_test = Test("paper_1d")
    # dann_test = Test('adv', threshold=True)
    
    # dann_test.config['dataset']['real_img_keep_size'] = True 

    test_thresholding(test_loader, threshold_value = 0.9 , dir_name_to_save="theshold_9_all")
    
    # for iter, data in enumerate(test_loader):
    #     noise, raw,name = data
    #     print(iter)
    #     # print(noise.shape)