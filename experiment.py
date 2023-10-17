import json, pickle
import os, math, glob
from traceback import print_tb
# from autoencoder_denoiser.dataloader_deprecated import get_datasets, get_real_img_dataset
from hsqc_dataset import  get_datasets, get_real_img_dataset
from model_factory import get_model , CDANLoss
from weakref import ref
import matplotlib.pyplot as plt
import numpy as np
import torch, copy, cv2
from tqdm import tqdm
from datetime import datetime
import shutil
import matplotlib.image
from sklearn.metrics import f1_score, average_precision_score, recall_score
from torch.utils.tensorboard import SummaryWriter

PRINT_TIME = False

os.system('nvidia-smi -L')
# os.system("lscpu")


class Experiment(object):
    def __init__(self, name):
        experiment_version = 'bitmap'
        f = open(f'/root/autoencoder_denoiser/configs_{experiment_version}/'+ name + '.json')
        # f = open(f'/root/autoencoder_denoiser/configs_baseline_selection/'+ name + '.json')
        # global config
        
        config = json.load(f)
        DEBUG = config.get("DEBUG")
        self.ROOT_STATS_DIR = f"./exps/results_{experiment_version}"
        if DEBUG:
            self.ROOT_STATS_DIR = f"./exps/results_debug"
            config['dataset']['batch_size'] = 4
        config['experiment_name'] = name
        self.config = config
        self.best_model = None

        if config is None:
            raise Exception("Configuration file doesn't exist: ", name)

        self.name=name
        self.__name = config['experiment_name']

        # make directory for this experiement
        self.__experiment_dir = os.path.join(self.ROOT_STATS_DIR, self.__name)
        self._test_samples_path = self.__experiment_dir+"/testing_sample_imgs"
        os.makedirs(self._test_samples_path, exist_ok=True)
        self._val_samples_path = self.__experiment_dir+"/val_sample_imgs"
        os.makedirs(self._val_samples_path, exist_ok=True)
        clist = [(0,"green"), (0.5,"white"), (1, "red")]
        self.custom_diff_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("_",clist)
        clist = [(0,"darkblue"), (0.5,"white"), (1, "darkred")]
        self.custom_HSQC_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("_",clist)

        # Load Datasets
        self.__train_loader, self.__val_loader, self.__test_loader = get_datasets(config)
        if config['model']['model_type'] == "Adv_UNet":
            # with open('/root/autoencoder_denoiser/dataset/imgs_as_array.pkl', 'rb') as f:
            #     self.__real_imgs  =  pickle.load(f)
            # self.__real_imgs = list(zip(*self.__real_imgs))[0]
            # self.real_img_index = 0
            self.__adv_criterion = CDANLoss(use_entropy=self.config['model']["use_entropy"])
            self.real_img_loader = get_real_img_dataset(config)
            self.__real_img_batch_iterator = loop_iterable(self.real_img_loader) 

        # Setup Experiment
        self.__epochs = config['experiment']['num_epochs']
        if DEBUG:
            self.__epochs = 2

        self.curr_iter = 0
        self.__current_epoch = 0
   
        self.__min_val_loss = float('inf')
        self.__learning_rate = config['experiment']['learning_rate']
        
        self.stop_progressing = 0
        self.__train_metric = Metric()
        self.__val_metric = Metric()
        self.__test_metric = Metric()
        self.writer = SummaryWriter(log_dir=self.__experiment_dir)

        # Init Model
        self.__model = get_model(config)
        self.__model = torch.nn.DataParallel(self.__model)
        
        if config['model']['model_type'] == 'filter':
            return None
        self.best_epoch = 0
        

        # Also assign GPU to device
        cuda_num = '0'
        self.device = torch.device(
            "cuda:{}".format(cuda_num) if torch.cuda.is_available() else "cpu"
        )
        
        print("model using cuda #{}".format(cuda_num))
        self.__model = self.__model.to(self.device)
        print("model finish moving")

        print(" choosing loss function and optimizer")
            
        if  config["experiment"]["loss_func"] == "MSE":
            self.__criterion = torch.nn.MSELoss()
        elif config["experiment"]["loss_func"] == "CrossEntropy":
            self.__criterion = torch.nn.CrossEntropyLoss() # edited
        else:
            raise Exception("what is your loss function??")
        
        self.__optimizer = torch.optim.Adam(self.__model.parameters(), lr = self.__learning_rate) # edited

        # add scheduler
        
        lr_step = config["experiment"]["lr_scheduler_step"]       
        self.lr_scheduler_type = config["experiment"]["lr_scheduler_type"]
        print(self.lr_scheduler_type)
        if self.lr_scheduler_type == "step":
            self.__lr_scheduler = torch.optim.lr_scheduler.StepLR(self.__optimizer, lr_step[0])
        elif self.lr_scheduler_type == "multi_step":
            self.__lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.__optimizer, lr_step)
        elif self.lr_scheduler_type == "criterion":
            self.__lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.__optimizer,'min')
        else: 
            self.__lr_scheduler = None 
        print(self.__lr_scheduler)

        self.__init_model()

        # Load Experiment Data if available
        
        # finetune with real img
        if self.config['model'].get("loading_path"):
            print("loading previous model to finetune by real noisy data")
            state_dict = torch.load(self.config['model'].get("loading_path"))
            # self.__model.module.Unet = torch.nn.DataParallel(self.__model.module.Unet, device_ids=self.model.device)
            self.__model.module.Unet.load_state_dict(state_dict['model'])
        self.__load_experiment()


    # Loads the experiment data if exists to resume training from last saved checkpoint.
    def __load_experiment(self):
        os.makedirs(self.ROOT_STATS_DIR, exist_ok=True)

        saved_model_path = os.path.join(os.path.join(self.ROOT_STATS_DIR, self.__name), 'latest_model.pt')
        
        if os.path.exists(saved_model_path):
            
 
                state_dict = torch.load(saved_model_path)
                '''{'model': model_dict, 
                      'optimizer': self.__optimizer.state_dict(),
                      'current_epoch':self.__current_epoch,
                      'min_val_loss':self.__min_val_loss,
                      'current_iter':self.curr_iter                      
                      }'''
                # state_dict = torch.load(os.path.join(os.path.join(ROOT_STATS_DIR, "adv"), 'latest_model.pt'))

                self.__model.module.load_state_dict(state_dict['model'])
                self.best_model = copy.deepcopy(self.__model)
                self.__optimizer.load_state_dict(state_dict['optimizer'])
                self.__current_epoch = state_dict['current_epoch']
                self.__min_val_loss= state_dict['min_val_loss']
                self.curr_iter = state_dict['current_iter']
                print("Successfully loaded previous model and states")
        else:
            for f in glob.glob(self.ROOT_STATS_DIR+'/'+self.__name+f"/events*"):
                os.remove(f)
            os.makedirs(self.__experiment_dir, exist_ok=True)

    def run(self):
        if self.config['model']['model_type'] == 'filter':
            return
        beginning_epoch = self.__current_epoch
        
        for i in range(beginning_epoch, self.__epochs):  # loop over the dataset multiple times
            if self.stop_progressing >= 10:
                print("Early stopped :)")
                break
            self.__current_epoch+=1
            print("epoch: ",self.__current_epoch)
            self.__train()
            val_loss = self.__val()
            
            if self.__lr_scheduler is not None:
                if self.lr_scheduler_type == "criterion":
                    self.__lr_scheduler.step(val_loss)
                else: 
                    self.__lr_scheduler.step()
        # self.plot_stats()


    def __train(self):
        self.__model.train()
        # temp
        # Iterate over the data, implement the training function
        if PRINT_TIME:
            import time
            start_time = time.time()
        for iter, data in enumerate(tqdm(self.__train_loader)):
            if PRINT_TIME and iter%20 == 0:
                print ("My program took", time.time() - start_time, "to run")
                start_time = time.time()
            self.__train_metric.reset()
            self.curr_iter += 1 
            raw, noise = self.__move_to_cuda(data)
            if self.config['model']['model_type'] == "Adv_UNet":
                # if self.config['dataset']['real_img_dataset_name']=="Chen":
                #     real_img = next(self.__batch_iterator).unsqueeze(1)
                # if self.config['dataset']['real_img_dataset_name']=="Byeol":
                #     end_index = self.real_img_index+self.config["dataset"]['batch_size']
                #     if end_index>len(self.__real_imgs):
                #         real_img = self.__real_imgs[self.real_img_index:]
                #         self.real_img_index = end_index-len(self.__real_imgs)
                #         real_img += self.__real_imgs[:self.real_img_index]                                    
                #     else:
                #         real_img = self.__real_imgs[self.real_img_index:end_index]
                #         self.real_img_index=end_index
                #     real_img = np.stack(real_img)
                #     real_img = torch.tensor(real_img).unsqueeze(1)
                 
                # for kk in self.real_img_loader:
                #     print(kk)
                #     pass   
                real_img = next(self.__real_img_batch_iterator)[0]
                if real_img.shape[0]< torch.cuda.device_count() :
                    real_img = torch.cat((real_img, next(self.__real_img_batch_iterator)[0].unsqueeze(1)))

                real_img = real_img.float().to(self.device)
            self.__optimizer.zero_grad()
            # print ("noise shape",noise.shape)        
            
            if self.config['model']['model_type'] == "Adv_UNet":
                adv_coeff = self.calc_coeff(self.curr_iter)
                # print(noise.shape, real_img.shape)
                prediction, domain_prediction, softmax_output = self.__model(x=noise, y=real_img,  coeff=adv_coeff, plain=False)
                # prediction = self.__model(x=noise, y=None,  coeff=adv_coeff)
            else:    
                prediction = self.__model.forward(noise)
            # prediction = prediction.type(torch.float32)
            # print(prediction.shape, raw.shape)
            ground_truth = raw
            if self.config["experiment"]["loss_func"] == "CrossEntropy":
                ground_truth = self.threshould_for_display(raw)
                
            loss = self.__criterion(prediction, ground_truth )
            
            # print("MSE loss is: ", loss)
            
            if self.config['model']['model_type'] == "Adv_UNet":
                # break
                dc_target = torch.cat((torch.ones(noise.shape[0]), torch.zeros(real_img.shape[0])), 0).float().to(self.device)
                                            # adv_loss = torch.nn.BCEWithLogitsLoss()(domain_prediction.squeeze(1), dc_target)
                adv_loss = self.__adv_criterion(ad_out=domain_prediction, softmax_output=softmax_output, coeff= adv_coeff,
                                                dc_target = dc_target)
                self.writer.add_scalar(f'train/adv_loss', adv_loss, self.curr_iter)   
                """Warning: if want to write adv_loss to tensorboard, 
                do it at the CDAN_LOSS module since the loss here is multiplied by the coeff"""
                # print('classify loss is: ', loss)
                loss = loss + 0.1*adv_loss
                # print("adv_loss is ",adv_loss)
             
                # print('total loss is: ', loss)
            #     pass
            prediction = torch.clip(prediction,0,1)
            with torch.no_grad():
                for i in range(len(raw)):
                    self.__train_metric.update(raw[i].detach(), noise[i].detach(), prediction[i].detach())
                    
            self.writer.add_scalar(f'train/loss', loss, self.curr_iter)        
            # losses += loss.item()
            self.__train_metric.avg(iter+1) # divided by batch_size
            self.__train_metric.write(self.writer, "train", self.curr_iter)
            
            loss.backward()
            self.__optimizer.step()

    def threshould_for_display(self, raw_img, threshold = 0.1):
        out_img = raw_img.detach().clone()
        shape = out_img.shape
        out_img = out_img.view((-1, shape[-2], shape[-1])) # to shape of batch*height*width (no channels)
        out_img[out_img>threshold]    = 2
        close_to_zeros = torch.logical_and(out_img<threshold, out_img>-1*threshold)
        out_img[close_to_zeros]   = 1
        out_img[out_img<-1*threshold] = 0
        return out_img.long()
            


    

    def __val(self):
        # print("validating stage")

        self.__model.eval()
        
        val_loss = 0
        
        with torch.no_grad():
            self.__val_metric.reset()
            for iter, data in enumerate(tqdm(self.__val_loader)):
                raw, noise = self.__move_to_cuda(data)
                prediction = self.__model.forward(noise)
                
            
                loss = self.val_step(iter, raw, noise,prediction)

                for i in range(len(raw)):
                    self.__val_metric.update(raw[i].detach(), noise[i].detach(), prediction[i].detach())
                val_loss += loss    
            val_loss = val_loss/(iter+1)
           
        self.writer.add_scalar(f'val/loss', val_loss, self.curr_iter)        
        self.__val_metric.avg((iter+1)*len(self.__val_loader)) # divided by num of all images
        self.__val_metric.write(self.writer, "val", self.curr_iter)
                
        if self.config['model']['model_type'] == "Adv_UNet":
            val_loss_on_real = 0
            with torch.no_grad():   
                for iter, data in enumerate(tqdm(self.real_img_loader )):
                    noise, raw = data    
                    if len(raw.shape)==3:   
                        raw, noise = raw.unsqueeze(1), noise.unsqueeze(1)
                    prediction = self.__model.forward(noise)
                    raw, noise = raw.to(self.device).float(), noise.to(self.device).float()
                    loss = self.val_step(iter, raw, noise, prediction, type="real")
                    val_loss_on_real += loss    
                val_loss_on_real = val_loss_on_real/(iter+1)
            self.writer.add_scalar(f'val/loss_on_real_imgs', val_loss_on_real, self.curr_iter)          


        if val_loss < self.__min_val_loss:
            self.__min_val_loss = val_loss
            self.__save_model()
            print("best model updated")
            self.best_epoch = self.__current_epoch
            self.best_model = copy.deepcopy(self.__model)
            self.stop_progressing = 0
        else:
            self.stop_progressing += 1
        return val_loss

    def val_step(self, iter, raw, noise,prediction, type = "synthesis"):
                
                # find loss
                # prediction = prediction.type(torch.float32)
        
        if self.config["experiment"]["loss_func"] == "CrossEntropy":
            ground_truth = self.threshould_for_display(raw)
            loss=self.__criterion(prediction,ground_truth )
            # modify in order to plot
            prediction = torch.argmax(prediction, dim=1)
            prediction = prediction - 1 
        else: 
            ground_truth = raw
            prediction = torch.clip(prediction,-1,1)
            loss=self.__criterion(prediction,ground_truth )
        
        
                # add adv loss !!!                    
                #draw sample pics
                # if self.__current_epoch% 15 ==0 and iter==0:
        if iter==0:
            if self.config['model']['model_type'] != 'filter' and self.config['model']['model_type'] != 'vanilla':
                noise_pic , prediction_pic, raw_pic = noise[0],prediction[0], raw[0]
            else: noise_pic , prediction_pic, raw_pic = noise,prediction, raw
            if self.config["experiment"]["loss_func"] == "CrossEntropy":
                prediction_pic = prediction
                 
            plt.clf()

            ax = plt.subplot(1, 3, 1)
            plt.tight_layout()
            ax.set_title('orig')
            # ax.axis('off')
            plt.imshow(raw_pic[0].cpu(),cmap=self.custom_HSQC_cmap, vmax=1, vmin=-1)

            ax = plt.subplot(1, 3, 2)
            plt.tight_layout()
            ax.set_title('noise')
            # ax.axis('off')
            plt.imshow(noise_pic[0].cpu(),cmap=self.custom_HSQC_cmap, vmax=1, vmin=-1)

            ax = plt.subplot(1, 3, 3)
            plt.tight_layout()
            ax.set_title('predicted')
            # ax.axis('off')

            plt.imshow(prediction_pic[0].cpu(),cmap=self.custom_HSQC_cmap, vmax=1, vmin=-1)

                    # print(os.path.join(self._val_samples_path, "epoch_{}_sample_images.png".format(str(self.__current_epoch))))
            plt.savefig(os.path.join(self._val_samples_path,f"epoch_{str(self.__current_epoch)}_{type}_images.png"))
            displayed = True
            plt.clf()
        
        return loss.item() 

    def test(self):
        print("testing stage")

        test_loss= 0

        self.__model.eval()
        
        displayed = 0

        with torch.no_grad():
            self.__test_metric.reset()
            for iter, data in enumerate(tqdm(self.__test_loader)):
                raw, noise = self.__move_to_cuda(data)
                prediction = self.best_model(noise).data
                
                loss = self.test_step(displayed, raw, noise, prediction)
                for i in range(len(raw)):
                    self.__test_metric.update(raw[i].detach(), noise[i].detach(), prediction[i].detach())
                test_loss += loss
                displayed += 1
            test_loss /= (iter+1)
           
            self.writer.add_scalar(f'test/loss', test_loss, self.curr_iter)        
            self.__test_metric.avg((iter+1)*len(self.__test_loader)) # divided by num of all images
            self.__test_metric.write(self.writer, "test", self.curr_iter)
            print("avg testing loss is ", test_loss)
            
        if self.config['model']['model_type'] == "Adv_UNet":
            with torch.no_grad(): 
                test_loss_on_real = 0  
                for iter, data in enumerate(tqdm(self.real_img_loader )):
                    noise, raw = data    
                    prediction = self.__model.forward(noise)
                    if len(raw.shape)==3:   
                        raw, noise = raw.unsqueeze(1), noise.unsqueeze(1)
                    raw, noise = raw.to(self.device).float(), noise.to(self.device).float()
                    loss = self.test_step(iter, raw, noise, prediction, type = "real")
                    test_loss_on_real += loss    
                test_loss_on_real = test_loss_on_real/(iter+1)
            self.writer.add_scalar(f'test/loss_on_real_imgs', test_loss_on_real, self.curr_iter)     

    def test_step(self, displayed_num, raw, noise, prediction, type = "synthesis"): 
        
        
        if self.config["experiment"]["loss_func"] == "CrossEntropy":
            ground_truth = self.threshould_for_display(raw)
            loss=self.__criterion(prediction,ground_truth )
            # modify in order to plot
            prediction = torch.argmax(prediction, dim=1)
            prediction = prediction - 1 
        else: 
            ground_truth = raw
            prediction = torch.clip(prediction,-1,1)
            loss=self.__criterion(prediction,ground_truth )
        


        if displayed_num<20:
            if self.config['model']['model_type'] != 'filter' and self.config['model']['model_type'] != 'vanilla':
                noise_pic , prediction_pic, raw_pic = noise[0],prediction[0], raw[0]
            else: noise_pic , prediction_pic, raw_pic = noise,prediction, raw
            if self.config["experiment"]["loss_func"] == "CrossEntropy":
                prediction_pic = prediction
                    
            if self.config["model"]['model_type'] == "JNet":
                noise_pic = noise_pic[0]
                    
            plt.clf()

            ax = plt.subplot(2, 2, 1)
            plt.tight_layout()
            ax.set_title('original')
            # ax.axis('off')
            plt.imshow(raw_pic[0].cpu(),cmap=self.custom_HSQC_cmap, vmax=1, vmin=-1)

            ax = plt.subplot(2, 2, 2)
            plt.tight_layout()
            ax.set_title('noise')
            # ax.axis('off')
            plt.imshow(noise_pic[0].cpu(),cmap=self.custom_HSQC_cmap, vmax=1, vmin=-1)

            ax = plt.subplot(2, 2, 3)
            plt.tight_layout()
            ax.set_title('predicted')
            # ax.axis('off')
            plt.imshow(prediction_pic[0].cpu(),cmap=self.custom_HSQC_cmap, vmax=1, vmin=-1)
                    
            ax = plt.subplot(2, 2, 4)
            plt.tight_layout()
            ax.set_title('difference')
            # ax.axis('off')
                    
                    # difference = prediction_pic[0].cpu()-raw_pic[0].cpu()
                    # difference = difference.float()/2 + 0.5
                    # print(difference)
            # print("prediction_pic.dtype: ", prediction_pic.dtype, "ground_truth.dtype: ", ground_truth.dtype)            
            difference = cv2.subtract(np.array(prediction_pic[0].cpu()), np.array(raw_pic[0].cpu()))
            plt.imshow(difference, cmap = self.custom_diff_cmap, vmax=1, vmin=-1)

                    # print(os.path.join(self._test_samples_path, f"sample_image{displayed}.png"))
            plt.savefig(os.path.join(self._test_samples_path, f"{type}_image{displayed_num}.png"))
            
            plt.clf()
        return loss    




    def __init_model(self):
        if torch.cuda.is_available():
            self.__model = self.__model.cuda().float()
            self.__criterion = self.__criterion.cuda()

    def __save_model(self):
        root_model_path = os.path.join(self.__experiment_dir, 'latest_model.pt')
        model_dict = self.__model.module.state_dict()
        state_dict = {'model': model_dict, 
                      'optimizer': self.__optimizer.state_dict(),
                      'current_epoch':self.__current_epoch,
                      'min_val_loss':self.__min_val_loss,
                      'current_iter':self.curr_iter                      
                      }
        torch.save(state_dict, root_model_path)

    def __move_to_cuda(self, data):
        raw, noise = data
        if type(noise) is list:
            raw= raw.to(self.device).float()
            noisy_sample, tessellated_noise = noise
            noise = noisy_sample.to(self.device).float(),  tessellated_noise.cuda().float()
        else:
            
            raw, noise = raw.to(self.device).float(), noise.to(self.device).float()
        return raw,noise  
            
    
    def calc_coeff(self, iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=1000.0):
        kick_in_iter = self.config['experiment'].get('adv_coeff_kick_in_iter')
        if kick_in_iter:
            coeff_param = kick_in_iter/10
        else : 
            coeff_param = 20.0
        return np.float(coeff_param* (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)


def loop_iterable(iterable):
    while True:
        # for i in  iterable: 
            # print(i)
            # yield i
        yield from iterable
        


# SNR helper:

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




def compute_SNR(raw, noisy_img): 
    signal_position= torch.where(raw!=0)
    # noise_position= torch.where(raw==0)
    # prediction_error = torch.sum( torch.abs(raw-noisy_img))
 
    avg_signal = torch.sum( torch.abs(raw))/len(signal_position[0])
    noise_std =  torch.std(noisy_img - raw)
    return (avg_signal/noise_std).item()


# def wSNR_increase(raw, noise, prediction):
#     orig_wSNR = compute_wSNR(raw, noise)
#     denoised_wSNR = compute_wSNR(raw, prediction)
#     return denoised_wSNR/orig_wSNR

def compute_metrics(raw, noise, prediction):
    # assert(raw.dim()==2)
    raw, noise, prediction = torch.abs(raw), torch.abs(noise), torch.abs(prediction)
    
    
    orig_SNR = compute_SNR(raw, noise)
    denoised_SNR = compute_SNR(raw, prediction)
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