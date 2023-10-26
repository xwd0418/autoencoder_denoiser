import os,sys
from glob import glob
import cv2, pickle
import torch, copy
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
# from data_preprocess import triangle_tessellate , expand
from PIL import Image

class HSQCDataset(Dataset):
    def __init__(self, split="train", config=None):
        
        seed = 10086
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        if config['dataset']['cleaned']:
            self.dir = "/root/data/SMILES_dataset/"
        else:
            self.dir = "/root/data/hyun_fp_data/hsqc_ms_pairs/"
        self.split = split
        self.config = config
        self.augment = config['dataset'].get("data_augment")
        self.augment = 1 if self.augment==None else int(self.augment)
        assert (self.augment >= 1)
        # self.orig_hsqc = os.path.join(self.dir, "data")
        # assert(os.path.exists(self.orig_hsqc))
        assert(split in ["train", "val", "test"])
        
        self.hsqc_path = 'HSQC_trimmed'
        if self.config['dataset'].get("large_input"):
            raise Exception("deprecated usaged of 1800*1200 size imgs")
            self.dir = "/data/hyun_fp_data/hsqc_ms_pairs/"
            self.hsqc_path = 'HSQC_plain_imgs_toghter_1800_1200'
            
        self.hsqc_files = list(os.listdir(os.path.join(self.dir, split, self.hsqc_path)))
        # self.HSQC_files = list(os.listdir(os.path.join(self.dir, split, "HSQC")))
        # assert (len(self.FP_files ) == (self.HSQC_files))

    def __len__(self): 
        if self.config.get("DEBUG"): 
            return 50
        return len(self.hsqc_files)*self.augment

    def __getitem__(self, i):
        
        
        
        file_index = i//self.augment

        raw_sample = torch.load(os.path.join(self.dir,  self.split, self.hsqc_path, self.hsqc_files[file_index]))
        # print(os.path.join(self.dir,  self.split, self.hsqc_path, self.hsqc_files[file_index]))
        raw_sample = np.array(raw_sample, dtype="float32")
        raw_sample = raw_sample[0]
        signal_enhance_factor =  self.config['dataset'].get('signal_enhance')
        if signal_enhance_factor:
            raw_sample = np.sign(raw_sample) * (np.abs(raw_sample)) ** signal_enhance_factor

        signal_enlarge_factor = self.config['dataset'].get('signal_enlarge')
        if signal_enlarge_factor:
            signal_position = np.where(raw_sample!=0)
            signal_enlarge = 1
            for i in range(0, signal_enlarge+1):
                for j in range(0, signal_enlarge+1):
                    if i==0 and j==0: continue
                    # here use index -2 -1 instead of 0 ,1 is because I have unsqueezed the image
                    new_position = np.clip(signal_position[0]+i, 0, raw_sample.shape[0]-1) ,  np.clip(signal_position[1]+j , 0, raw_sample.shape[1]-1)  
                    raw_sample[new_position] += raw_sample[signal_position]
        raw_sample = np.clip(raw_sample, -1. , 1.)
            
        noise_factor = random.uniform(self.config["dataset"]["noise_factor"][0], self.config["dataset"]["noise_factor"][1])
   
        # add noise based on provided noise type
        if self.config["dataset"]["noise_type"] == "random":
            self.config["dataset"]["noise_type"] = random.choice(["gaussian", "white" ])
        elif self.config["dataset"]["noise_type"] == "gaussian":
            noisy_sample = raw_sample + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=raw_sample.shape)
        elif self.config["dataset"]["noise_type"] == "white":    
            noisy_sample = raw_sample + noise_factor * np.random.uniform(low=-1.0, high=1.0, size=raw_sample.shape)
        elif self.config["dataset"]["noise_type"] == "t1": 
            noisy_sample = add_t1_noise(raw_sample, self.config)

            white_noise_chance_bound = self.config['dataset'].get('white_noise_chance_bound')
            if white_noise_chance_bound is None : white_noise_chance_bound = [0.005, 0.2]
            low_bound, up_bound = white_noise_chance_bound[0],white_noise_chance_bound[1]
            chance_shown = np.random.uniform(low_bound, up_bound)
            shown_positions = np.random.binomial(1, chance_shown,size=raw_sample.shape)
            half_half = np.random.binomial(1, 0.5,size=raw_sample.shape)
            half_half[half_half==0]=-1
            shown_positions = shown_positions * half_half
            noisy_sample +=  noise_factor * shown_positions

        else:
            raise Exception("unkown type of noise {}".format(self.config["dataset"]["noise_type"]))
       
        noisy_sample = np.clip(noisy_sample, -1. , 1.)


        if self.config['model']['model_type'] != 'filter' and self.config['model']['model_type'] != 'vanilla':
            raw_sample = np.expand_dims(raw_sample, axis=0)
            noisy_sample = np.expand_dims(noisy_sample, axis=0)

        if self.config["dataset"]["pre-filtered"]:
            noisy_sample = np.array([[[filtering(float(k)) for k in j] for j in i] for i in noisy_sample])
        
        # if self.config["dataset"]["tessellate"]:
            
        #     selected_noisy_sample = np.array([[[selecting(float(k)) for k in j] for j in i] for i in noisy_sample])
        #     tessellated_noise = triangle_tessellate( selected_noisy_sample[0], self.config["dataset"]["upscale"])
        #     tessellated_noise = np.expand_dims(tessellated_noise, axis=0)
        #     # tessellated_noise = np.expand_dims(tessellated_noise, axis=0)
        #     # noisy_sample = np.stack((expand(noisy_sample), tessellated_noise), axis=0)
            
        #     ### using low resolution of tessellation 
        #     if self.config['model']['model_type'] == "UNet_2": 
        #         concat = np.concatenate((noisy_sample, tessellated_noise))    
        #         if self.config['dataset']['absolute']:
        #             raw_sample,concat = np.abs(raw_sample), np.abs(concat)            
        #         return raw_sample, concat
        #     ### using Jnet 
        #     else:
        #         if self.config['dataset']['absolute']:
        #             raw_sample, noisy_sample, tessellated_noise = np.abs(raw_sample), np.abs(noisy_sample), np.abs(tessellated_noise)
        #         return raw_sample, (noisy_sample, tessellated_noise)
        if self.config['dataset']['absolute']:
            raw_sample, noisy_sample = np.abs(raw_sample), np.abs(noisy_sample)
            
        upscale_factor = self.config['dataset'].get('signal_upscale')
        if upscale_factor!=None:
            # upscaled_shape = (raw_sample.shape[2]*upscale_factor,raw_sample.shape[1]*upscale_factor)
            # raw_sample = cv2.resize(raw_sample[0], upscaled_shape , interpolation = cv2.INTER_LINEAR) 
            # noisy_sample = cv2.resize(noisy_sample[0], upscaled_shape , interpolation = cv2.INTER_LINEAR) 
            raw_sample = np.repeat(raw_sample, upscale_factor, axis=1)
            raw_sample = np.repeat(raw_sample, upscale_factor, axis=2)
            noisy_sample =  np.repeat(noisy_sample, upscale_factor, axis=1)
            noisy_sample =  np.repeat(noisy_sample, upscale_factor, axis=2)
        return raw_sample, noisy_sample
    


class RealNoiseDataset_Chen(Dataset):
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
    
class RealNoiseDataset_Byeol(Dataset):
    def __init__(self, config, range_low = 0, range_high = float('inf'), 
                 show_name=False,
                 ) -> None:
        super().__init__() 
        self.show_name = show_name
        if config['dataset']['super_noisy']:          
            self.data_folder_name = "resized_super_noisy_1"
            print("using Byeol's real imgs: super noisy")
        else:
            self.data_folder_name = 'resized_real_imgs_bitmap'
            print("using Byeol's real imgs: normal noise")
        # bitmap = "_bitmap"
        # if config['dataset'].get('countor_map'):
        #     bitmap=""
        
        if config['dataset'].get('real_img_keep_size'):
            self.data_folder_name = self.data_folder_name.replace("resized", "orig_size")
        self.real_img_data_dir = f'/root/autoencoder_denoiser/dataset/{self.data_folder_name}/'
        self.paths =  glob(self.real_img_data_dir+"*")
        self.paths = [path for path in self.paths if 
                      range_low <= int(path.split("_")[-1].split(".")[0]) <= range_high]

    def  __len__(self):
        return len(self.paths)
        
    def __getitem__(self, index):
        
        loaded_data = np.load(self.paths[index])
        noise, ground_truth = loaded_data['noise'], loaded_data['ground_truth']
        noise = cv2.resize(noise, (120, 180))
        ground_truth = cv2.resize(ground_truth, (120, 180))
        if not self.show_name:
            return (np.expand_dims(noise,0), np.expand_dims(ground_truth,0))
        return (np.expand_dims(noise,0), np.expand_dims(ground_truth,0), loaded_data['name'])
            
# class MultiStageRealNoiseDataset(Dataset):
#     def __init__(self, noise_level, split) -> None:
#         super().__init__()         
#         with open(f'/root/autoencoder_denoiser/dataset/save_real_imgs_in_stages/stage_{noise_level}.pkl', 'rb') as f:
#             imgs =  pickle.load(f)
#         random.shuffle(imgs)
        
            
#     def  __len__(self):
#         return len(self.imgs)
        
#     def __getitem__(self, index):
#         return self.imgs[index]
            

def get_datasets(config):
    
    DEBUG = config.get("DEBUG")
    num_workers = 0 if DEBUG else 16
    shuffle=config["dataset"]['shuffle']
    batch = config["dataset"]['batch_size']
    pin_mem = False
    persistent_workers = True
    train_loader = DataLoader(HSQCDataset("train", config), batch_size=batch, shuffle=shuffle, num_workers=num_workers,
                              pin_memory=pin_mem, persistent_workers=persistent_workers)
    val_loader = DataLoader(HSQCDataset("val",config), batch_size=batch, shuffle=shuffle, num_workers=num_workers,
                              pin_memory=pin_mem, persistent_workers=persistent_workers)
    test_loader = DataLoader(HSQCDataset("test",config), batch_size=batch, shuffle=shuffle, num_workers=num_workers,
                              pin_memory=pin_mem, persistent_workers=persistent_workers)
    return train_loader, val_loader , test_loader

def get_real_img_dataset(config):
    DEBUG = config.get("DEBUG")
    num_workers = 0 if DEBUG else 16
    batch = config["dataset"]['batch_size']
    if config['dataset'].get('real_img_keep_size') :
        batch = 2
    shuffle=config["dataset"]['shuffle']
    if config['dataset']['real_img_dataset_name']=="Chen":
        print("using Chen's real imgs")
        return DataLoader(RealNoiseDataset_Chen(config), batch_size=min(16,batch), shuffle=shuffle, num_workers=num_workers)
    if config['dataset']['real_img_dataset_name']=="Byeol":
        
        return DataLoader(RealNoiseDataset_Byeol(config), batch_size=min(16,batch), shuffle=shuffle, num_workers=num_workers)
            


"""helper functioner to gerneate noise"""

def selecting(x):
    if x>=0.6: return 1
    return 0

def filtering(x):
    if x>=0.6: return x
    return 0


def add_t1_noise(img, config):
    """_summary_

    Args:
        img (_2d numpy array, e.g. size of 180*120_): input image without noise
        config (dict): some hyperparameters to determine how the noise will be

    """
    height, width = img.shape[0], img.shape[1]
    p_streak_range = config['dataset'].get('streak_prob')  # probability of generating streak noise
    p_streak = np.random.uniform(low=p_streak_range[0], high=p_streak_range[1])
    
    noisy_img = copy.deepcopy(img)
    cross_noise, cross_points = generate_cross_noise(img, config)
    
    # vertical streak noises
    noisy_columns = cross_points[1]
    for col in noisy_columns:
        if np.random.binomial(1, p_streak):
            noise_rate = np.random.binomial(height, np.random.uniform(low=0.2, high=0.7))/ height
            noise =  np.random.binomial(1, noise_rate, height)
            
            # this is only need it the image contains different kinds of H-C bound,
            # i.e. input contains both positive and negative values
            if np.random.binomial(1, 0.5):  
                noise*=-1
            noise_factor = random.uniform(config["dataset"]["noise_factor"][0], config["dataset"]["noise_factor"][1])
            noisy_img[:,col] += noise*noise_factor
   
    # horizontal streak noises
    noisy_rows = cross_points[0]
    for row in noisy_rows:
        if np.random.binomial(1, p_streak/2): # division by 2 is because there are more vertical streaks
            noise_rate = np.random.binomial(width, np.random.uniform(low=0.2, high=0.7))/ width
            noise =  np.random.binomial(1, noise_rate, width)
            # same as above : this is only need it the image contains different kinds of H-C bound,
            # i.e. input contains both positive and negative values
            if np.random.binomial(1, 0.5): 
                noise*=-1
            noise_factor = random.uniform(config["dataset"]["noise_factor"][0], config["dataset"]["noise_factor"][1])
            noisy_img[row] += noise*noise_factor
   
    # same as above : this is only need it the image contains different kinds of H-C bound,
     # i.e. input contains both positive and negative values
    if np.random.binomial(1, 0.5): 
        cross_noise*=-1
    noise_factor = random.uniform(config["dataset"]["noise_factor"][0], config["dataset"]["noise_factor"][1])
    noisy_img = noisy_img + cross_noise*noise_factor
        
    return noisy_img
        

def generate_cross_noise(img, config):
    noise_probability_range = config['dataset']['cross_prob']
    noise_probability = np.random.uniform(low=noise_probability_range[0], high=noise_probability_range[1])
    output_noise = np.zeros(img.shape)
    points =  np.array(np.where(img>0))
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