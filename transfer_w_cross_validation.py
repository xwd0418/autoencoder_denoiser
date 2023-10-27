# ''''''
'''
This is used for the purpose of cross-validation, with or without a pretrained(from my mimicked data) model
'''

import json
import sys
import os
import pickle
import random
import numpy as np
import torch
import copy
import cv2
from glob import glob
from tqdm import tqdm
from model_factory import get_model, CDANLoss
import matplotlib.image
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from utils import display_pics
from hsqc_dataset import RealNoiseDataset_Byeol
from utils import display_pics, compute_metrics
from torchmetrics.functional import precision_recall


# class CrossValidateDataset(Dataset):
'''
deprecated , because when cross validation, we want to holdout a compound each time
'''
#     def __init__(self, paths) -> None:
#         super().__init__()
#         self.paths = paths

#     def __len__(self):
#         return len(self.paths)

#     def __getitem__(self, index):
#         loaded_data = np.load(self.paths[index])
#         noise, ground_truth = loaded_data['noise'], loaded_data['ground_truth']
#         # noise = cv2.resize(noise, (120, 180))
#         # ground_truth = cv2.resize(ground_truth, (120, 180))
#         return np.expand_dims(noise, 0),  np.expand_dims(ground_truth, 0)

# class DatasetFromList(Dataset): 
#     def __init__(self, imgs) -> None:
#         super().__init__() 
#         self.imgs = imgs

#     def  __len__(self):
#         return len(self.imgs)
        
#     def __getitem__(self, index):
#         return self.imgs[index]
    
    
    
class DenoiseExp(object):
    def __init__(self, name) -> None:
        config_dir = '/root/autoencoder_denoiser/configs_cross_validation'
        f = open(f'{config_dir}/' + name + '.json')
        config = json.load(f)
        self.config = config

        self.__experiment_dir = '/root/autoencoder_denoiser/exps/cross_validation/'+name
        os.system(f'rm -r {self.__experiment_dir}')
        self.device = torch.device("cuda:0")
        self.__criterion = torch.nn.MSELoss()
        self.__criterion = self.__criterion.cuda()
        self.epoch = 50

    def load_model(self, loading_path):
        if loading_path:
            print('loading weights')
            self.saved_model_path = loading_path
            state_dict = torch.load(self.saved_model_path)
            self.__model.module.load_state_dict(state_dict['model'])
            self.best_model = copy.deepcopy(self.__model)
        else:
            print('train from sratch')

    # def partition_dataset(self, k):
    '''
    deprecated , because when cross validation, we want to holdout a compound each time
    '''
    #     """partitioning dataset in k different ways for cross-validation

    #     Args:
    #         k (int): num of cross-validation

    #     Yields:
    #         train/val/test loaders 
    #     """

    #     dataset = RealNoiseDataset_Byeol(self.config).paths
    #     # random.shuffle(dataset)

    #     for i in range(k):
    #         eighty_percent = int(len(dataset)*0.8)
    #         nighty_percent = int(len(dataset)*0.9)

    #         train_partition = dataset[:eighty_percent]
    #         val_partition = dataset[eighty_percent:nighty_percent]
    #         test_partition = dataset[nighty_percent:]
    #         batch_size = self.config['dataset']['batch_size']
    #         print("training data amount: ", len(train_partition))
    #         train_loader = DataLoader(CrossValidateDataset( train_partition), batch_size=batch_size, shuffle=True,
    #                                   persistent_workers=True,  num_workers=16, pin_memory = True
    #                                   )
    #         print('finishing preparing train loader')
    #         val_loader = DataLoader(CrossValidateDataset(val_partition), batch_size=batch_size, shuffle=True,
    #                                 persistent_workers=True,  num_workers=16, pin_memory = True
    #                                   )
    #         test_loader = DataLoader(CrossValidateDataset(test_partition), batch_size=batch_size, shuffle=True,
    #                                  persistent_workers=True,  num_workers=16, pin_memory = True
    #                                   )
    #         yield train_loader, val_loader, test_loader
    #         split_point = len(dataset)//k
    #         dataset = dataset[split_point:]+dataset[:split_point]


    def generate_loaders(self,k):
        with open('/root/autoencoder_denoiser/dataset/all_names.pkl', 'rb') as f:
            all_names =  pickle.load(f)


        for _ in range(k):
            eighty_percent = int(len(all_names)*0.8)
            nighty_percent = int(len(all_names)*0.9)
            # print(eighty_percent, nighty_percent, len(all_names))
            train_partition = all_names[:eighty_percent]
            val_partition = all_names[eighty_percent:nighty_percent]
            test_partition = all_names[nighty_percent:]
            
            train_loaders , val_loader ,test_loader = self.get_loaders_from_partition(train_partition,val_partition,test_partition)
            yield train_loaders, val_loader, test_loader
            split_point = len(all_names)//k
            all_names = all_names[split_point:]+all_names[:split_point]
                
    def get_loaders_from_partition(self,train_partition,val_partition,test_partition):
        self.img_parent_dir = '/root/autoencoder_denoiser/dataset/group_by_name_and_stage'
        
        val_loader = self.get_loader(val_partition)
        test_loader = self.get_loader(test_partition)
        train_loaders = []
        
        num_stages = self.config['experiment'].get("num_stage")
        if num_stages == None:
            num_stages = 1
        noise_level_range = range(1, num_stages+1 )
        for iter, noise_level in enumerate(noise_level_range):
            if iter == len(noise_level_range)-1:
                train_loaders.append(self.get_loader(train_partition))                
            else:
                train_loaders.append(self.get_loader(train_partition, loader_noise_level=noise_level))
        
        return train_loaders, val_loader, test_loader
        

    def get_loader(self, train_or_val_or_test_partition, loader_noise_level=None):
        # print(' buiding ....')
        # print(train_or_val_or_test_partition)
        data_list = []
        for name in train_or_val_or_test_partition:
            coumpound_dir = os.path.join(self.img_parent_dir, name)
            coumpound_imgs_paths = sorted(glob(coumpound_dir+"/*"))
            for noise_img in coumpound_imgs_paths[:-1]: # last one is ground truth
                            # none noise level means validating or testing
                img_noise_level = int(noise_img.split('.')[0].split('_')[-1])
                in_range = loader_noise_level == None or img_noise_level>=loader_noise_level
             
                
                if in_range:
                    noise, ground_truth = np.load(noise_img), np.load(coumpound_imgs_paths[-1])
                    if self.config['dataset'].get('absolute'):
                        noise, ground_truth = np.abs(noise), np.abs(ground_truth)
                    data_list.append((np.expand_dims(noise,axis=0), np.expand_dims(ground_truth,axis=0)))
        loader_output = DataLoader((data_list), batch_size=self.config['dataset']['batch_size'], shuffle=True)
        return loader_output
    
    def run(self, k_fold=1):
        loaders_generator = self.generate_loaders(k=k_fold)
        self.k_fold = k_fold
        for k in range(k_fold):
            # init model
            self.__model = get_model(self.config)
            self.__model = torch.nn.DataParallel(self.__model)
            print('moving model to cuda')
            if torch.cuda.is_available():
                self.__model = self.__model.cuda().float()
            self.load_model(self.config['loading_path'])

            # init training configs

            self.__learning_rate = self.config['experiment']['learning_rate']
            self.__optimizer = torch.optim.Adam(
                self.__model.parameters(), lr=self.__learning_rate)  # edited
            self.__lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.__optimizer, 'min')

            curr_experiment_dir = self.__experiment_dir+f'/fold_{k+1}'
            self.writer = SummaryWriter(log_dir=curr_experiment_dir)
            self._test_samples_path = curr_experiment_dir+"/testing_sample_imgs"
            os.makedirs(self._test_samples_path, exist_ok=True)
            
            print(f"cross validation {k+1}/{k_fold}")
            train_loaders, val_loader, test_loader = next(loaders_generator)

            self.stop_progressing = 0
            curr_iter = 0
            self.__min_val_loss = float("inf")
            for train_loader_index, train_loader in enumerate(train_loaders):
                self.stop_progressing=0
                
                
                for epoch in tqdm(range(self.epoch)):
                    # early stop

                    if self.stop_progressing >= 10:
                        print("Early stopped :)")
                        break

                    # train
                    for iter, data in enumerate((train_loader)):
                        # print(f"training {iter}")
                        noise, ground_truth = data
                        ground_truth, noise = ground_truth, noise
                        ground_truth, noise = ground_truth.to(
                            self.device).float(), noise.to(self.device).float()
                        self.__optimizer.zero_grad()
                        prediction = self.__model.forward(noise)
                        
                        if self.config["experiment"]["loss_func"] == "BCE":
                            ground_truth = torch.where(ground_truth > 0, 1.0, 0.0)
                        loss = self.__criterion(prediction, ground_truth)
                        self.writer.add_scalar(f'train/loss', loss, curr_iter)
                        loss.backward()
                        
                        # computing metrics
                        orig_SNR, denoised_SNR, SNR_incr = compute_metrics(ground_truth, noise, prediction)
                        
                        
                        prediction = torch.where(prediction != 0, 1.0, 0.0)
                        ground_truth = torch.where(ground_truth != 0, 1.0, 0.0)
                        precision ,recall = precision_recall(prediction,ground_truth.int())
                        f1_score = 2*precision*recall/(precision+recall)
                        
                        self.writer.add_scalar(f'train/precision', precision, curr_iter)
                        self.writer.add_scalar(f'train/recall', recall, curr_iter)
                        self.writer.add_scalar(f'train/orig_SNR', orig_SNR, curr_iter)
                        self.writer.add_scalar(f'train/denoised_SNR', denoised_SNR, curr_iter)
                        self.writer.add_scalar(f'train/SNR_incr', SNR_incr, curr_iter)
                        self.__optimizer.step()
                        curr_iter += 1
                        
                    # val
                    self._val_samples_path = curr_experiment_dir+"/val_sample_imgs/training_loader_{train_loader_index}/"
                    os.makedirs(self._val_samples_path, exist_ok=True)
                    val_loss = 0
                    orig_SNR, denoised_SNR, SNR_incr = 0,0,0
                    precision, recall = 0, 0
                    with torch.no_grad():
                        for iter, data in enumerate((val_loader)):

                            noise, ground_truth = data
                            ground_truth, noise = ground_truth, noise
                            ground_truth, noise = ground_truth.to(self.device).float(), noise.to(self.device).float()
                            prediction = self.__model.forward(noise)
                            if self.config["experiment"]["loss_func"] == "BCE":
                                ground_truth = torch.where(ground_truth > 0, 1.0, 0.0)
                            val_loss += self.__criterion(prediction, ground_truth)
                            # updating orig_SNR, denoised_SNR, SNR_incr
                            orig_SNR, denoised_SNR, SNR_incr = [sum(x) for x in zip( (orig_SNR, denoised_SNR, SNR_incr) , \
                                                                                        compute_metrics(ground_truth, noise, prediction))] 
                            prediction = torch.where(prediction != 0, 1.0, 0.0)
                            ground_truth = torch.where(ground_truth != 0, 1.0, 0.0)
                            precision_cur ,recall_cur = precision_recall(prediction,ground_truth.int())
                            precision += precision_cur
                            recall += recall_cur
                            
                            
                        self.writer.add_scalar( f'val/loss', val_loss/(iter+1), epoch)
                  
                        # f1_score = 2*precision*recall/(precision+recall)
                        
                        self.writer.add_scalar(f'train/precision', precision/(iter+1), curr_iter)
                        self.writer.add_scalar(f'train/recall', recall/(iter+1), curr_iter)
                        self.writer.add_scalar(f'val/orig_SNR', orig_SNR/(iter+1), curr_iter)
                        self.writer.add_scalar(f'val/denoised_SNR', denoised_SNR/(iter+1), curr_iter)
                        self.writer.add_scalar(f'val/SNR_incr', SNR_incr/(iter+1), curr_iter)
                        self.__lr_scheduler.step(val_loss)
                    # update early stop
                    if val_loss < self.__min_val_loss:
                        self.__min_val_loss = val_loss
                        torch.save(self.__model, curr_experiment_dir +
                                "/best_model.pt")
                        # print("best model updated")
                        self.best_model = copy.deepcopy(self.__model)
                        self.stop_progressing = 0
                    else:
                        self.stop_progressing += 1

                    if epoch % 20 == 0:

                        display_pics(noise[0,0].cpu(), prediction[0,0].cpu(), ground_truth[0,0].cpu(), save_path=self._val_samples_path+f"epoch_{epoch}")
            # test
            test_loss = 0
            orig_SNR, denoised_SNR, SNR_incr = 0,0,0
            precision, recall = 0, 0
            with torch.no_grad():
                for iter, data in enumerate((test_loader)):
                    noise, ground_truth = data
                    ground_truth, noise = ground_truth, noise
                    ground_truth, noise = ground_truth.to(self.device).float(), noise.to(self.device).float()
                    prediction = self.__model.forward(noise)
                    if self.config["experiment"]["loss_func"] == "BCE":
                        ground_truth = torch.where(ground_truth > 0, 1.0, 0.0)
                    test_loss += self.__criterion(prediction, ground_truth)
                    orig_SNR, denoised_SNR, SNR_incr = [sum(x) for x in zip( (orig_SNR, denoised_SNR, SNR_incr) , \
                                                                                        compute_metrics(ground_truth, noise, prediction))] 
                    prediction = torch.where(prediction != 0, 1.0, 0.0)
                    ground_truth = torch.where(ground_truth != 0, 1.0, 0.0)
                    precision_cur ,recall_cur = precision_recall(prediction,ground_truth.int())
                    precision += precision_cur
                    recall += recall_cur
                            
                            
                for i in range(len(prediction)):
                    display_pics(noise[0,0].cpu(), prediction[0,0].cpu(), ground_truth[0,0].cpu(), save_path=self._test_samples_path+f"/num_{i}")
                    
                self.writer.add_scalar(f'test/loss', test_loss/(iter+1), 0)
                self.writer.add_scalar(f'train/precision', precision/(iter+1), curr_iter)
                self.writer.add_scalar(f'train/recall', recall/(iter+1), curr_iter)
                self.writer.add_scalar(f'test/orig_SNR', orig_SNR/(iter+1), curr_iter)
                self.writer.add_scalar(f'test/denoised_SNR', denoised_SNR/(iter+1), curr_iter)
                self.writer.add_scalar(f'test/SNR_incr', SNR_incr/(iter+1), curr_iter)


if len(sys.argv) > 1:
    name = sys.argv[1]
else:
    # default name
    name = "??"
os.system('nvidia-smi -L')
seed = 3405
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
exp = DenoiseExp(name)
exp.run(k_fold=10)
