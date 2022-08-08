
import json
import os
from weakref import ref
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import shutil
import matplotlib.image
from glob import glob
import torch
from model_factory import UNet

orig_img_dir = "/home/wangdong/autoencoder_denoiser/dataset/real_noise"
new_img_dir = orig_img_dir+"_binary_array"
for img_folder in glob(new_img_dir+"/*/"):
    for img_path in glob(img_folder+"/*"):
        img = np.load(img_path)
        plt.imshow(img, cmap="Greys")
        break
    break
  
model_name = "improved_t1"
experiment_dir=f"/home/wangdong/autoencoder_denoiser/experiment_data/{model_name}/"
state_dict = torch.load(os.path.join(experiment_dir, 'latest_model.pt'))

model = UNet(1,1,True) 
model = torch.nn.DataParallel(model)
model.load_state_dict(state_dict['model'])

saved_dir = "/home/wangdong/autoencoder_denoiser/denoised_with_{}/".format(model_name)
os.makedirs(saved_dir,exist_ok=True)
import cv2
orig_img_dir = "/home/wangdong/autoencoder_denoiser/dataset/real_noise"
new_img_dir = orig_img_dir+"_binary_array"

clist = [(0,"green"), (0.25,"white"), (0.75, "black"), (1, "red")]
# red means noise
custom_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("_",clist)
binary_map = matplotlib.colors.LinearSegmentedColormap.from_list("_",[(0, "white"), (1, "black")])
# model.to("cpu")
model.eval()
count = 0
torch.cuda.empty_cache()
device = torch.device("cuda:0")
model.to(device)

def change_dilation(model):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            change_dilation(module)
        if isinstance(module, torch.nn.Conv2d):
            module.dilation = (3,3)
            # padding = module.padding[0]+(module.kernel_size[0]-1)/2
            if module.kernel_size[0]==3:
                module.padding = (3,3)

# change_dilation(model)


for img_folder in glob(new_img_dir+"/*/"):
    for img_path in glob(img_folder+"/*"):
        # try:
            count+=1
            img = np.load(img_path)
            # plt.imshow(img, cmap="Greys")
            
            img_input = torch.tensor(img).unsqueeze(0).unsqueeze(0).float().to(device)
            try:
                denoised = model.forward(img_input)
            except :
                img = cv2.resize(img.astype("uint8"), (img.shape[0]//2, img.shape[1]//2))
                img_input = torch.tensor(img).unsqueeze(0).unsqueeze(0).float().to(device)
                denoised = model.forward(img_input)
            denoised = torch.clip(denoised.round(),0,1)
            denoised_result = denoised[0,0].cpu().detach().numpy()
            difference = img*3 - denoised_result 
            plt.imshow(difference, cmap = custom_cmap, vmax=3, vmin=-1)
            plt.savefig(saved_dir+f"{count}_comparison.png")

            
            plt.imshow(denoised_result, cmap=binary_map,  vmax=1, vmin=0)
            plt.savefig(saved_dir+f"{count}_denoised.png")
            # del img_input,denoised
            torch.cuda.empty_cache()
            # print(torch.cuda.memory_allocated(0)//1024)
        # except:
        #     print(img_path)
        #     print(count)
        #     exit()
            if count > 30 : 
                break
    break
    
        