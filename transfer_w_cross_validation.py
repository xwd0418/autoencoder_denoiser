# ''''''
'''
This is used for the purpose of cross-validation, with or without a pretrained(from my mimicked data) model
it also records avg precision and recall across all folds 
'''

import json
import sys
import os
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
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('xtick', labelsize=15) 
matplotlib.rc('ytick', labelsize=15) 

from utils import display_pics, compute_metrics, generate_loaders
from torchmetrics.functional import precision_recall
from loss_functions import get_loss_func



THRESHOLD_FOR_PRECISION_RECALL = 0.02

    
class DenoiseExp(object):
    def __init__(self, name, config = None) -> None:
        if config == None:
            config_dir = '/root/autoencoder_denoiser/configs_cross_val_precision_recall'
            f = open(f'{config_dir}/' + name + '.json')
            config = json.load(f)
        self.config = config

        self.__experiment_dir = '/root/autoencoder_denoiser/exps/cross_validation_with_precision_recall_v2/'+name
        os.system(f'rm -r {self.__experiment_dir}')
        self.device = torch.device("cuda:0")
        self.__criterion = get_loss_func(config)
        self.__criterion = self.__criterion.cuda()
        self.epoch = 200
        if config.get('DEBUG'):
            self.epoch = 4

    def load_model(self, loading_path):
        if loading_path:
            print('loading weights')
            self.saved_model_path = loading_path
            state_dict = torch.load(self.saved_model_path)
            self.__model.module.load_state_dict(state_dict['model'])
            self.best_model = copy.deepcopy(self.__model)
        else:
            print('train from sratch')


    def run(self, k_fold=1):
        loaders_generator = generate_loaders(k=k_fold, config = self.config)
        self.k_fold = k_fold
        test_results = []
        
        total_levels_of_noises = self.config['experiment'].get("num_stage")
        if total_levels_of_noises is None:
            total_levels_of_noises = 1
        k_fold_SNRs, k_fold_precisions, k_fold_recalls = [[] for _ in range(total_levels_of_noises)],\
                                                            [[] for _ in range(total_levels_of_noises)],\
                                                                [[] for _ in range(total_levels_of_noises)],
        
        
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
            all_loaders = next(loaders_generator)
            # print()
            # print(all_loaders)
            # print(type(all_loaders))
            self.stop_progressing = 0
            curr_iter = 0
            self.__min_val_loss = float("inf")
            
            HSQCs_seen_before = set()
            # use this set to record what HSQC images have been seen before in lower noise levels
            # so we can make sure to have images of this current noise level 
            
            
            # for each noise level
            for loader_index, (train_loader, val_loader, test_loader) in enumerate(zip(*all_loaders)):
                
                self.stop_progressing=0
                
                for epoch in tqdm(range(self.epoch)):
                    # early stop
                    if self.stop_progressing >= 20:
                        print("Early stopped :)")
                        break

                    # train
                    self.train_sample_path = curr_experiment_dir+f"/train_sample_imgs/traini_noise_level_{loader_index+1}/"
                    os.makedirs(self.train_sample_path, exist_ok=True)
                    for iter, data in enumerate((train_loader)):
                        ground_truth, noise  = data
                        ground_truth, noise = ground_truth.to(self.device).float(), noise.to(self.device).float()
                        self.__optimizer.zero_grad()
                        prediction = self.__model.forward(noise)
                        
                        # if self.config["experiment"]["loss_func"] == "BCE":
                        #     ground_truth = torch.where(ground_truth > 0, 1.0, 0.0)
                        loss = self.__criterion(prediction, ground_truth)
                        self.writer.add_scalar(f'train/loss', loss, curr_iter)
                        loss.backward()
                        
                        # computing metrics
                        orig_SNR, denoised_SNR, SNR_incr = compute_metrics(ground_truth, noise, prediction)
                        
                        
                        # if self.config["experiment"]["loss_func"] == "BCE":
                        #     prediction = torch.where(prediction > 0, 1.0, 0.0)
                        # else:
                        if self.config["experiment"]["loss_func"] == "MSE":
                            prediction = torch.where(torch.abs(prediction) > THRESHOLD_FOR_PRECISION_RECALL, 1.0, 0.0)
                        # ground_truth = torch.where(ground_truth != 0, 1.0, 0.0)
                        precision ,recall = precision_recall(prediction,ground_truth.int())
                        # f1_score = 2*precision*recall/(precision+recall)
                        
                        self.writer.add_scalar(f'train/precision', precision, curr_iter)
                        self.writer.add_scalar(f'train/recall', recall, curr_iter)
                        self.writer.add_scalar(f'train/orig_SNR', orig_SNR, curr_iter)
                        self.writer.add_scalar(f'train/denoised_SNR', denoised_SNR, curr_iter)
                        self.writer.add_scalar(f'train/SNR_incr', SNR_incr, curr_iter)
                        self.__optimizer.step()
                        curr_iter += 1
                        
                        if epoch % 20 == 0:

                            display_pics(noise[0,0].cpu(), prediction[0,0].cpu(), ground_truth[0,0].cpu(),
                                         save_path=self.train_sample_path+f"epoch_{epoch}", config=self.config)
                        
                    # val
                    self._val_samples_path = curr_experiment_dir+f"/val_sample_imgs/training_loader_{loader_index}/"
                    os.makedirs(self._val_samples_path, exist_ok=True)
                    val_loss = 0
                    orig_SNR, denoised_SNR, SNR_incr = 0,0,0
                    precision, recall = 0, 0
                    with torch.no_grad():
                        for iter, data in enumerate((val_loader)):

                            ground_truth, noise  = data
                            
                            ground_truth, noise = ground_truth.to(self.device).float(), noise.to(self.device).float()
                            prediction = self.__model.forward(noise)
                            # if self.config["experiment"]["loss_func"] == "BCE":
                            #     ground_truth = torch.where(ground_truth > 0, 1.0, 0.0)
                            val_loss += self.__criterion(prediction, ground_truth)
                            # updating orig_SNR, denoised_SNR, SNR_incr; fancy way of multi-self-increment 
                            orig_SNR, denoised_SNR, SNR_incr = [sum(x) for x in zip( (orig_SNR, denoised_SNR, SNR_incr) , \
                                                                                        compute_metrics(ground_truth, noise, prediction))] 
                            # if self.config["experiment"]["loss_func"] == "BCE":
                            #     prediction = torch.where(prediction > 0, 1.0, 0.0)
                            # else:
                            if self.config["experiment"]["loss_func"] == "MSE":
                                prediction = torch.where(torch.abs(prediction) > THRESHOLD_FOR_PRECISION_RECALL, 1.0, 0.0)
                            # ground_truth = torch.where(ground_truth != 0, 1.0, 0.0)
                            precision_cur ,recall_cur = precision_recall(prediction,ground_truth.int())
                            precision += precision_cur
                            recall += recall_cur
                            
                            
                        self.writer.add_scalar( f'val/loss', val_loss/(iter+1), epoch)
                  
                        # f1_score = 2*precision*recall/(precision+recall)
                        
                        self.writer.add_scalar(f'val/precision', precision/(iter+1), curr_iter)
                        self.writer.add_scalar(f'val/recall', recall/(iter+1), curr_iter)
                        self.writer.add_scalar(f'val/orig_SNR', orig_SNR/(iter+1), curr_iter)
                        self.writer.add_scalar(f'val/denoised_SNR', denoised_SNR/(iter+1), curr_iter)
                        self.writer.add_scalar(f'val/SNR_incr', SNR_incr/(iter+1), curr_iter)
                        self.__lr_scheduler.step(val_loss)
                    # update early stop
                    if val_loss < self.__min_val_loss:
                        self.__min_val_loss = val_loss
                        # torch.save(self.__model, curr_experiment_dir + "/best_model.pt")
                        # print("best model updated")
                        self.best_model = copy.deepcopy(self.__model)
                        self.stop_progressing = 0
                    else:
                        self.stop_progressing += 1

                    if epoch % 20 == 0:

                        display_pics(noise[0,0].cpu(), prediction[0,0].cpu(), ground_truth[0,0].cpu(), 
                                     save_path=self._val_samples_path+f"epoch_{epoch}", config=self.config)
                        
                        
                        
                # test
                test_loss = 0
                orig_SNR, denoised_SNR, SNR_incr = 0,0,0
                precision, recall = 0, 0
                self.__model = self.best_model
                with torch.no_grad():
                    for iter, data in enumerate((test_loader)):
                        
                        ground_truth, noise  = data
                        ground_truth, noise = ground_truth.to(self.device).float(), noise.to(self.device).float()
                        prediction = self.__model.forward(noise)
                        # if self.config["experiment"]["loss_func"] == "BCE":
                        #     ground_truth = torch.where(ground_truth > 0, 1.0, 0.0)
                        
                        test_loss += self.__criterion(prediction, ground_truth)
                        
                        orig_SNR_curr, denoised_SNR_curr, SNR_incr_curr = compute_metrics(ground_truth, noise, prediction)
                        orig_SNR+=orig_SNR_curr
                        denoised_SNR += denoised_SNR_curr
                        SNR_incr += SNR_incr_curr
                        
                        # if self.config["experiment"]["loss_func"] == "BCE":
                        #     prediction = torch.where(prediction > 0, 1.0, 0.0)
                        # else:
                        if self.config["experiment"]["loss_func"] == "MSE":
                            prediction = torch.where(torch.abs(prediction) > THRESHOLD_FOR_PRECISION_RECALL, 1.0, 0.0)
                        # ground_truth = torch.where(ground_truth != 0, 1.0, 0.0)
                        precision_cur ,recall_cur = precision_recall(prediction,ground_truth.int())
                        precision += precision_cur
                        recall += recall_cur
                                
                        k_fold_SNRs[loader_index].append(orig_SNR_curr  )
                        k_fold_precisions[loader_index].append(precision_cur.cpu().detach().numpy()   )
                        k_fold_recalls[loader_index].append(recall_cur.cpu().detach().numpy()   )
                        
                        for i in range(len(prediction)):
                            if (torch.sum(noise)).item() in HSQCs_seen_before:
                                continue # not the current noise level
                            display_pics(noise[i,0].cpu(), prediction[i,0].cpu(), ground_truth[i,0].cpu(), 
                                        save_path=self._test_samples_path+f"/noise_level_{loader_index+1}_num_{iter}", config=self.config)
                        HSQCs_seen_before.add((torch.sum(noise)).item())
                        
                    self.writer.add_scalar(f'test/loss', test_loss/(iter+1), curr_iter)
                    self.writer.add_scalar(f'test/precision', precision/(iter+1), curr_iter)
                    self.writer.add_scalar(f'test/recall', recall/(iter+1), curr_iter)
                    self.writer.add_scalar(f'test/orig_SNR', orig_SNR/(iter+1), curr_iter)
                    self.writer.add_scalar(f'test/denoised_SNR', denoised_SNR/(iter+1), curr_iter)
                    self.writer.add_scalar(f'test/SNR_incr', SNR_incr/(iter+1), curr_iter)

                    test_results.append([test_loss/(iter+1),  precision/(iter+1), recall/(iter+1),
                                        orig_SNR/(iter+1),denoised_SNR/(iter+1),SNR_incr/(iter+1)])
                    

            
        # this model completed k-fold cross calidation, output txt files and plots 
        out_file = open(self.__experiment_dir+'/output.txt', 'w')
        all_results = list(zip(*test_results))
        metric_names = ['loss', 'precision', 'recall', 'orig_SNR', 'denoised_SNR',  'SNR_incr']
        for result, metric_name in zip(all_results, metric_names):         
            out_file.write(f'average {metric_name}: {sum(result)/k_fold}\n')
        out_file.close()
        
        # precision recall of different noise levels 
        for i in range(len(k_fold_SNRs)):
            stage_i_SNRs =  k_fold_SNRs[i]
            stage_i_precisions =  k_fold_precisions[i]
            stage_i_recalls =  k_fold_recalls[i]

            plt.figure()
            plt.title(f"Precision and Recalls of noise level {i+1} for {k_fold}-fold cross validation", fontsize=26)     
                
            plt.scatter(stage_i_SNRs, stage_i_precisions, color='r', label='precisions') 
            plt.scatter(stage_i_SNRs, stage_i_recalls, color='g', label='recalls') 
            plt.legend(fontsize=16)
            
            # Naming the x-axis, y-axis and the whole graph 
            plt.xlabel("SNR of images before denoising", fontsize=24) 
            plt.ylabel("Magnitude", fontsize=24) 
            
            plt.savefig(self.__experiment_dir+f'/precision_recall_noise_level{i+1}.png')
            plt.clf()
            
        
if __name__ == "__main__":
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
