import matplotlib.pyplot as plt
import matplotlib.image
import cv2, numpy as np
import torch

""" Part 1
plotting helper
"""
clist = [(0,"darkblue"), (0.5,"white"), (1, "darkred")]
custom_HSQC_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("_",clist)
diff_clist = [(0,"green"), (0.5,"white"), (1, "red")]
custom_diff_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("_",diff_clist)
plt.rcParams["figure.figsize"] = (16,20)

def display_pics(noise_pic, prediction_pic, raw_pic, title_name=None, save_path=None):
    '''
    display noisey pic, prediction, clean pic, and difference repestively
    input should be a 2D array/tensor
    '''
   
    
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
    difference = cv2.subtract(np.array(prediction_pic), np.array(raw_pic))
    plt.imshow(difference, cmap = custom_diff_cmap, vmax=1, vmin=-1)

    plt.colorbar()
    


                    
    if save_path:
        plt.savefig(save_path)
        plt.clf()
        plt.figure()
        plt.close()
                
    else:
        plt.figure()
        
        
        


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