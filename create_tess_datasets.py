from scipy.spatial import Delaunay
import cv2, torch
import numpy as np
from tqdm import tqdm


def triangle_tessellate(img1, points=None, scale_factor=1):
    # width = int(img1.shape[1] * scale_factor )
    # height = int(img1.shape[0] * scale_factor )
    dim = (180, 120)
    # img1 = cv2.resize(np.array(img1,dtype='uint8'), dim, interpolation = cv2.INTER_AREA)

    # find points
    if points==None :
        points = np.array(np.where(img1!=0)).T 
    
    #find triangles to draw
    try :
        tri = Delaunay(points)        
    except:
        return torch.tensor(img1).float().unsqueeze(0)
          
    triangles = points[tri.simplices[:]]
    
    triangles = triangles[:,:,[1,0]]     # swap column to match original image
    triangles = scale_factor*triangles   # for better resolution 

    # Blue color in BGR
    color = (1)
    
    # Line thickness of 2 px
    thickness = 1
    
    # Using cv2.polylines() method
    # Draw a Blue polygon with
    # thickness of 1 px

    output_img = np.zeros(dim, np.uint8)

    for triangle in triangles:
        output_img = cv2.polylines(np.array(output_img), [triangle],
                        True, color, thickness)
        
    ret_val = torch.tensor(output_img).float().unsqueeze(0)
    return ret_val

import os, torch
dir = "/root/data/hyun_fp_data/hsqc_ms_pairs/"
splits = ['train', 'val', 'test']
for split in splits:
    os.makedirs(f"{dir}{split}/HSQC_plain_imgs/", exist_ok = True)
    os.makedirs(f"{dir}{split}/HSQC_tessellation_imgs/", exist_ok = True)

    
    FP_files = list(os.listdir(os.path.join(dir, split, "HSQC")))
    for ite,file in enumerate(tqdm(FP_files)):
        coord = torch.load(os.path.join(dir, split, "HSQC", file))
        coord[:,0] = torch.clamp(coord[:,0], 0, 179)
        coord[:,1] = torch.clamp(coord[:,1]*10, 0, 119)  
        coord[:,1] = 119 - coord[:,1]

        pos = coord[coord[:,2]>0]
        neg = coord[coord[:,2]<0]     
        img_pos = torch.zeros((180,120)) 
        img_neg = torch.zeros((180,120)) 
        img_pos[tuple(torch.round(pos[:,:2]).long().T)] = pos[:,2]
        img_neg[tuple(torch.round(neg[:,:2]).long().T)] = torch.abs(neg[:,2])
        
        img_pos = cv2.normalize(np.array(img_pos), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        img_neg = cv2.normalize(np.array(img_neg), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        img_pos= cv2.flip(img_pos, 1)
        img_neg= cv2.flip(img_neg, 1)
        img_neg, img_pos = torch.tensor(img_neg), torch.tensor(img_pos)      
        
        tess_pos = triangle_tessellate(np.array(img_pos))
        tess_neg = triangle_tessellate(np.array(img_neg))
        
        img_pos = img_pos.unsqueeze(0)
        img_neg = img_neg.unsqueeze(0)
        
        imgs = torch.cat((img_pos,img_neg))
        torch.save(imgs, f"{dir}{split}/HSQC_plain_imgs/{file}")
        tessellations = torch.cat((tess_pos,tess_neg))
        torch.save(tessellations, f"{dir}{split}/HSQC_tessellation_imgs/{file}")
imgs.shape