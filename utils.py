import matplotlib.pyplot as plt
import matplotlib.image
import cv2, numpy as np
import torch
from torchmetrics.functional import precision_recall


""" Part 1
plotting helper
"""
clist = [(0,"darkblue"), (0.5,"white"), (1, "darkred")]
custom_HSQC_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("_",clist)
diff_clist = [(0,"green"), (0.5,"white"), (1, "red")]
custom_diff_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("_",diff_clist)
plt.rcParams["figure.figsize"] = (16,20)

@torch.no_grad()
def display_pics(noise_pic, prediction_pic, raw_pic, title_name=None, save_path=None ,  config= None):
    '''
    display noisey pic, prediction, clean pic, and difference repestively
    input should be a 2D array/tensor
    '''
    if config and config['experiment']['loss_func'] != "MSE":
        prediction_pic = torch.sigmoid(prediction_pic)
    prediction_pic = prediction_pic.detach().numpy()
    raw_pic = raw_pic.detach().numpy()
    
    if title_name:
        plt.title(title_name)
   
    ax = plt.subplot(2, 2, 1)
    plt.tight_layout()
    ax.set_title('original',fontsize=18)
                    # ax.axis('off')
    plt.imshow(raw_pic,cmap=custom_HSQC_cmap, vmax=1, vmin=-1)
    plt.colorbar()
                    
    ax = plt.subplot(2, 2, 2)
    plt.tight_layout()
    ax.set_title('noise',fontsize=18)
                    # ax.axis('off')
    plt.imshow(noise_pic,cmap=custom_HSQC_cmap, vmax=1, vmin=-1)
    plt.colorbar()
                    
    ax = plt.subplot(2, 2, 3)
    plt.tight_layout()
    ax.set_title('predicted',fontsize=18)
                    # ax.axis('off')
    plt.imshow(prediction_pic,cmap=custom_HSQC_cmap, vmax=1, vmin=-1)
    plt.colorbar()
                    
    ax = plt.subplot(2, 2, 4)
    plt.tight_layout()
    ax.set_title('difference', fontsize=18)
                    # ax.axis('off')
                    
                    # difference = prediction_pic-raw_pic
                    # difference = difference.float()/2 + 0.5
                    # print(difference)
    difference = cv2.subtract(prediction_pic, raw_pic)
    plt.imshow(difference, cmap = custom_diff_cmap, vmax=1, vmin=-1)

    plt.colorbar()
                    
    if save_path:
        plt.savefig(save_path)
        plt.clf()
        plt.figure()
        plt.close()
                
    else:
        plt.figure()
        

@torch.no_grad()        
def display_precision_recalls(model, loader, title_name=None, save_path=None ,  config= None):
    device = torch.device("cuda:0")
    noise_pic_SNRs = []
    precisions = []
    recalls = []
    for iter, data in enumerate((loader)): 
        ground_truth, noise =  data
        ground_truth, noise = ground_truth.to(device).float(), noise.to(device).float()
        prediction = model.forward(noise)
        noise_pic_SNRs.add(__compute_SNR(ground_truth, noise))
        
        precision ,recall = precision_recall(prediction,ground_truth.int())
        precisions.append(precision)
        recalls.append(recall)
        
    plt.figure()
    if title_name:
        plt.title(title_name)     
        
    plt.plot(noise_pic_SNRs, precisions, color='r', label='precisions') 
    plt.plot(noise_pic_SNRs, recalls, color='g', label='recalls') 
    
    # Naming the x-axis, y-axis and the whole graph 
    plt.xlabel("SNR of images before denoising") 
    plt.ylabel("Magnitude") 
    
    if save_path:
        plt.savefig(save_path)
        plt.clf()
        
    return noise_pic_SNRs, precisions , recalls

''' part 2
SNR helper:
'''

class Metric():
    
    def init(self):
        self.reset()
        
    def reset(self):
        self.snr_orig  = 0
        self.snr_denoised = 0
        self.snr_inc  = 0
        self.wsnr_orig = 0
        self.wsnr_denoised = 0
        self.wsnr_inc  = 0
    
    def update(self, raw, noise, prediction):
        #  , orig_wSNR, denoised_wSNR,wSNR_inc
        orig_SNR, denoised_SNR,  SNR_inc = \
                    compute_metrics(torch.squeeze(raw,0), torch.squeeze(noise,0),torch.squeeze(prediction,0))
                    # compute_metrics(torch.squeeze(raw,0), torch.squeeze(noise,0),torch.squeeze(prediction,0), raw_noise_threadshold = 0, topk_k=3)
        self.snr_orig  += orig_SNR
        self.snr_denoised += denoised_SNR
        self.snr_inc += SNR_inc
        # self.wsnr_orig += orig_wSNR
        # self.wsnr_denoised += denoised_wSNR
        # self.wsnr_inc  += wSNR_inc
        
    def write(self, writer, mode, curr_iter):
        writer.add_scalar(f'{mode}/SNR_orig', self.snr_orig, curr_iter) 
        writer.add_scalar(f'{mode}/SNR_denoised', self.snr_denoised, curr_iter) 
        writer.add_scalar(f'{mode}/SNR_inc', self.snr_inc, curr_iter) 
        # writer.add_scalar(f'{mode}/wSNR_orig', self.wsnr_orig, curr_iter) 
        # writer.add_scalar(f'{mode}/wSNR_denoised', self.wsnr_denoised, curr_iter) 
        # writer.add_scalar(f'{mode}/wSNR_inc', self.wsnr_inc, curr_iter) 
        
    def avg(self, total_num):
        self.snr_orig  /= total_num
        self.snr_denoised /= total_num
        self.snr_inc  /= total_num
        # self.wsnr_orig /= total_num
        # self.wsnr_denoised /= total_num
        # self.wsnr_inc  /= total_num




def __compute_SNR(raw, noisy_img): 
    signal_position= torch.where(raw!=0)
    avg_signal = torch.sum( torch.abs(raw))/len(signal_position[0])
    noise_std =  torch.std(noisy_img - raw)
    return (avg_signal/noise_std).item()


# def wSNR_increase(raw, noise, prediction):
#     orig_wSNR = compute_wSNR(raw, noise)
#     denoised_wSNR = compute_wSNR(raw, prediction)
#     return denoised_wSNR/orig_wSNR

def compute_metrics(raw, noise, prediction):
    '''
    return orig_SNR, denoised_SNR, SNR increasment ratio
    '''
    
    # assert(raw.dim()==2)
    raw, noise, prediction = torch.abs(raw), torch.abs(noise), torch.abs(prediction)
    orig_SNR = __compute_SNR(raw, noise)
    denoised_SNR = __compute_SNR(raw, prediction)
    # orig_wSNR = compute_wSNR(raw, noise)
    # denoised_wSNR = compute_wSNR(raw, prediction)
    SNR_inc = denoised_SNR/orig_SNR
    # wSNR_inc = denoised_wSNR/orig_wSNR
    
    return orig_SNR, denoised_SNR, SNR_inc #, orig_wSNR, denoised_wSNR,  wSNR_inc


    

    
# def compute_metrics(raw, noise, prediction, raw_noise_threadshold=0.05, topk_k = 4):
#     noise_position= torch.where(raw<=raw_noise_threadshold)
#     signal_position= torch.where(raw>raw_noise_threadshold)
#     topk_k=min(topk_k,len(signal_position[0]) )
    
    
#     orig_SNR = torch.max(noise)/torch.std(noise[noise_position])
    
#     # print("torch std",torch.std(prediction[noise_position]), prediction[noise_position])
#     noised_std = max(torch.std(prediction[noise_position]).item(), 0.00001)
#     denoised_SNR = torch.max(prediction)/noised_std
    
#     # print("noise", noised_std, "snr", denoised_SNR)
#     # print("prediction's noise:",torch.max(prediction[noise_position]) )
#     # print("torch.std(prediction[noise_position])", torch.std(prediction[noise_position]))
#     SNR_inc = denoised_SNR/orig_SNR
    
#     orig_wSNR = torch.mean(torch.topk(noise[signal_position], k=topk_k, largest=False).values.clip_(0,1)) /torch.std(noise[noise_position])
#     denoised_wSNR = torch.mean(torch.topk(torch.where(prediction.double()>0.0, prediction.double(), 99.9), k=topk_k, largest=False).values)/noised_std
#     # print(torch.topk(torch.where(prediction.double()>0.0, prediction.double(), 99.9), k=4, largest=False).values, noised_std)
#     wSNR_inc = denoised_wSNR/orig_wSNR
    
#     # print(orig_SNR, denoised_SNR, orig_wSNR, denoised_wSNR, SNR_inc, wSNR_inc)
#     return orig_SNR.item(), denoised_SNR.item(), orig_wSNR.item(), denoised_wSNR.item(), SNR_inc.item(), wSNR_inc.item()




'''part 3, real imgs loader help
    It is used to give loaders with different levels of noises
'''

from glob import glob
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from hsqc_dataset import RealNoiseDataset_Byeol
import os, pickle

def generate_loaders(k, config):
    """generate loaders for k-fold cross validation, toghether with curriculum training

    Args:
        k (int): k-fold
        num_stages (_type_): it could be None, which means no stage, so loaders will include everything

    Yields:
        _type_: loaders
    """
    with open('/root/autoencoder_denoiser/dataset/all_names.pkl', 'rb') as f:
        all_names =  pickle.load(f)

    num_stages = config['experiment'].get("num_stage")
    for _ in range(k):
        eighty_percent = int(len(all_names)*0.8)
        nighty_percent = int(len(all_names)*0.9)
        # print(eighty_percent, nighty_percent, len(all_names))
        train_partition = all_names[:eighty_percent]
        val_partition = all_names[eighty_percent:nighty_percent]
        test_partition = all_names[nighty_percent:]
        
        train_loaders , val_loaders ,test_loaders = get_loaders_from_partition(
                    train_partition,val_partition,test_partition, config, num_stages
        )
        yield train_loaders, val_loaders, test_loaders
        split_point = len(all_names)//k
        all_names = all_names[split_point:]+all_names[:split_point]
            
def get_loaders_from_partition(train_partition,val_partition,test_partition, config, num_stages):
    
    if num_stages == None:
        
        train_loaders = get_loader(train_partition, config)
        val_loader = get_loader(val_partition, config)
        test_loader = get_loader(test_partition, config, is_test_loader=True)
        return [train_loaders], [val_loader], [test_loader]
    
    train_loaders, val_loader, test_loader = [], [], []
    
    for i in range(num_stages):
        noise_level = i+1
        if noise_level == num_stages:
            # return loaders that include evcerything
            train_loader_all, val_loader_all, test_loader_all =  get_loaders_from_partition(train_partition,val_partition,test_partition, config, None) 
            train_loaders += train_loader_all
            val_loader    += val_loader_all
            test_loader   += test_loader_all
        else:    
            train_loaders.append(get_loader(train_partition, config, noise_level=noise_level))
            val_loader   .append(get_loader(val_partition,   config, noise_level=noise_level))
            test_loader  .append(get_loader(test_partition,  config, noise_level=noise_level, is_test_loader=True))
    
    return train_loaders, val_loader, test_loader
    

def get_loader( compound_names, config, noise_level=None, is_test_loader = False):
    """return a dataloader of real images

    Args:
        compound_names (list of strings): names of coumpounds
        noise_level (int): _description_. Defaults to None.  
        
        In the files, Larger level means cleaner imgs. 
        Here, level should be in range of 1 ~ 15, 1 means the cleanest imgs, 15 means the noisiest

    """
    # print(' buiding ....')
    # print(train_or_val_or_test_partition)
    img_parent_dir = '/root/autoencoder_denoiser/dataset/group_by_name_and_stage'
    data_list = []
    for name in compound_names:
        coumpound_dir = os.path.join(img_parent_dir, name)
        coumpound_imgs_paths = sorted(glob(coumpound_dir+"/*"))
        coumpound_imgs_level_of_noise_and_paths = [  (int(path.split('.')[0].split('_')[-1]),path)  \
                    for path in coumpound_imgs_paths[:-1] ] # last one is ground truth
        # It should look like [(0, dataset/xxx.np), (1, dataset/xxx.np), .......]
        
        # determine how many data should include for the loader with such level_of_noise specification
        coumpound_imgs_level_of_noise_and_paths.sort(reverse=True)
        if noise_level == None:
            num_data_to_output = len(coumpound_imgs_level_of_noise_and_paths)
        else:
            num_data_to_output = min(len(coumpound_imgs_level_of_noise_and_paths), noise_level)
            
        for _level, noise_img in   coumpound_imgs_level_of_noise_and_paths[:num_data_to_output]  :
            
                noise, ground_truth = np.load(noise_img), np.load(coumpound_imgs_paths[-1])
                if config['dataset'].get('absolute'):
                    noise, ground_truth = np.abs(noise), np.abs(ground_truth)
                data_list.append((np.expand_dims(ground_truth,axis=0), np.expand_dims(noise,axis=0)))
    batch_size = 1 if is_test_loader else config['dataset']['batch_size']
    loader_output = DataLoader((data_list), batch_size=batch_size, shuffle=True)
    return loader_output
