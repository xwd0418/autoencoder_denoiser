import os, torch
from scipy.spatial import Delaunay
import cv2
from tqdm import tqdm
import numpy as np

dir = "/root/data/hyun_fp_data/hsqc_ms_pairs/"
splits = ['train', 'val', 'test']
for split in splits:
    os.makedirs(f"/data/hyun_fp_data/hsqc_ms_pairs/{split}/HSQC_plain_imgs_toghter_1800_1200/", exist_ok = True)

    
    FP_files = list(os.listdir(os.path.join(dir, split, "HSQC")))
    for ite,file in enumerate(tqdm(FP_files)):

        coord = torch.load(os.path.join(dir, split, "HSQC", file))
        coord[:,0] = torch.clamp(coord[:,0]*10, 0, 1799)
        coord[:,1] = torch.clamp(coord[:,1]*100, 0, 1199)  
        coord[:,1] = 119 - coord[:,1]

        # pos = coord[coord[:,2]>0]
        # neg = coord[coord[:,2]<0]     
        # img_pos = torch.zeros((180,120)) 
        # img_neg = torch.zeros((180,120)) 
        # img_pos[tuple(torch.round(pos[:,:2]).long().T)] = pos[:,2]
        # img_neg[tuple(torch.round(neg[:,:2]).long().T)] = torch.abs(neg[:,2])
        img = torch.zeros((1800,1200)) 
        img[tuple(torch.round(coord[:,:2]).long().T)] = torch.abs(coord[:,2])
        
        img = cv2.normalize(np.array(img), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        # img_neg = cv2.normalize(np.array(img_neg), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        # img_neg, img_pos = torch.tensor(img_neg), torch.tensor(img_pos) 
        # tess_img = triangle_tessellate(np.array(img))     
        img = torch.tensor(img).unsqueeze(0)
        # tess_img = torch.tensor(tess_img)
        
        
        torch.save(img, f"/data/hyun_fp_data/hsqc_ms_pairs/{split}/HSQC_plain_imgs_toghter_1800_1200/{file}")
        # torch.save(tess_img, f"{dir}{split}/HSQC_tessellation_imgs_toghter/{file}")

        
