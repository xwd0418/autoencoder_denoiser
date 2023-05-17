import json, pickle, random, sys 
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
from torch.utils.tensorboard import SummaryWriter



os.system('nvidia-smi -L')


class Experiment(object):
    def __init__(self, name):
        self.experiment_version = 'configs_autoencoder_configs'
        self.config_files = glob.glob(f'/root/autoencoder_denoiser/{self.experiment_version}/{name}/*')
        self.config_files.sort()
        f = open(self.config_files[0])
        # f = open(f'/root/autoencoder_denoiser/configs_baseline_selection/'+ name + '.json')
        # global config
        
        config = json.load(f)
        DEBUG = config.get("DEBUG")
        self.DEBUG = DEBUG
        self.ROOT_STATS_DIR = f"./exps/results_{self.experiment_version}"
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
        
        clist = [(0,"green"), (0.5,"white"), (1, "red")]
        self.custom_diff_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("_",clist)
        clist = [(0,"darkblue"), (0.5,"white"), (1, "darkred")]
        self.custom_HSQC_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("_",clist)

        # Load Datasets
        # self.__train_loader, self.__val_loader, self.__test_loader = get_datasets(config)
        
        self.loader_generater = self.produce_generater()
        # Setup Experiment
        self.__epochs = config['experiment']['num_epochs']
        if DEBUG:
            self.__epochs = 2

        self.curr_iter = 0
        self.__current_epoch = 0
   
        self.__learning_rate = config['experiment']['learning_rate']
        
        self.stop_progressing = 0
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
        # self.__load_experiment()

    def produce_generater(self):
        for config_file in (self.config_files):
            print(config_file)
            f = open(config_file)
            config = json.load(f)
            config["DEBUG"] = self.DEBUG
            curr_train_loader, curr_val_loader, curr_test_loader = get_datasets(config)
            stage_name = config_file.split("/")[-1].split(".")[0]
            val_samples_path = self.__experiment_dir+"/val_sample_imgs/"+stage_name
            os.makedirs(val_samples_path, exist_ok=True)
            test_samples_path = self.__experiment_dir+"/testing_sample_imgs/"+stage_name
            os.makedirs(test_samples_path, exist_ok=True)
            yield curr_train_loader, curr_val_loader, curr_test_loader , val_samples_path, test_samples_path

        
        
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
        self.update_loaders()
        if self.config['model']['model_type'] == 'filter':
            return
        beginning_epoch = 0
        
        #re-initialize training configs:
        self.__optimizer = torch.optim.Adam(self.__model.parameters(), lr = self.__learning_rate) # edited
        self.__lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.__optimizer,'min')
        self.__min_val_loss = float("inf")
        
        self.stop_progressing = 0
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

    def update_loaders(self):
        self.__train_loader, self.__val_loader, self.__test_loader, \
            self._val_samples_path, self._test_samples_path          = next(self.loader_generater)
        
        # self.plot_stats()


    def __train(self):
        self.__model.train()
        # temp
        # Iterate over the data, implement the training function
        for iter, data in enumerate(tqdm(self.__train_loader)):
            self.curr_iter += 1 
            raw, noise = self.__move_to_cuda(data)
            self.__optimizer.zero_grad()
            # print ("noise shape",noise.shape)        
                
            prediction = self.__model.forward(noise)
            # prediction = prediction.type(torch.float32)
            # print(prediction.shape, raw.shape)
            ground_truth = raw
            if self.config["experiment"]["loss_func"] == "CrossEntropy":
                ground_truth = self.threshould_for_display(raw)
                
            loss = self.__criterion(prediction, ground_truth )
            
            # print("MSE loss is: ", loss)
            
            prediction = torch.clip(prediction,0,1)
            
            self.writer.add_scalar(f'train/loss', loss, self.curr_iter)        
            # losses += loss.item()
            
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
            for iter, data in enumerate(tqdm(self.__val_loader)):
                raw, noise = self.__move_to_cuda(data)
                prediction = self.__model.forward(noise)
                
            
                loss = self.val_step(iter, raw, noise,prediction)

                val_loss += loss    
            val_loss = val_loss/(iter+1)
           
        self.writer.add_scalar(f'val/loss', val_loss, self.curr_iter)


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
            for iter, data in enumerate(tqdm(self.__test_loader)):
                raw, noise = self.__move_to_cuda(data)
                prediction = self.best_model(noise).data
                
                loss = self.test_step(displayed, raw, noise, prediction)
                test_loss += loss
                displayed += 1
            test_loss /= (iter+1)
           
            self.writer.add_scalar(f'test/loss', test_loss, self.curr_iter)        
            print("avg testing loss is ", test_loss)
            
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
        root_model_path = os.path.join(self._val_samples_path ,  'latest_model.pt')
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
            


if __name__ == "__main__":
    exp_name = 'default'
    
    seed = 10086
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if len(sys.argv) > 1:
        exp_name = sys.argv[1]

    print("Running Experiment: ", exp_name)
    exp = Experiment(exp_name)
    for _ in exp.config_files:
        exp.run()
        exp.test() 


