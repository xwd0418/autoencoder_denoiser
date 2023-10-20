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
from tqdm import tqdm
from model_factory import get_model, CDANLoss
import matplotlib.image
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from utils import display_pics
from hsqc_dataset import RealNoiseDataset_Byeol
from utils import display_pics, compute_metrics



class CrossValidateDataset(Dataset):
    def __init__(self, paths) -> None:
        super().__init__()
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        loaded_data = np.load(self.paths[index])
        return np.expand_dims(loaded_data['noise'], 0),  np.expand_dims(loaded_data['ground_truth'], 0)


class DenoiseExp(object):
    def __init__(self, name) -> None:
        config_dir = '/root/autoencoder_denoiser/configs_cross_validation'
        f = open(f'{config_dir}/' + name + '.json')
        config = json.load(f)
        self.config = config

        self.__experiment_dir = '/root/autoencoder_denoiser/exps/cross_validation/'+name
        self.device = torch.device("cuda:0")
        self.__criterion = torch.nn.MSELoss()
        self.__criterion = self.__criterion.cuda()
        self.epoch = 1000

    def load_model(self, loading_path):
        if loading_path:
            print('loading weights')
            self.saved_model_path = loading_path
            state_dict = torch.load(self.saved_model_path)
            self.__model.module.load_state_dict(state_dict['model'])
            self.best_model = copy.deepcopy(self.__model)
        else:
            print('train from sratch')

    def partition_dataset(self, k):
        """partitioning dataset in k different ways for cross-validation

        Args:
            k (int): num of cross-validation

        Yields:
            train/val/test loaders 
        """

        dataset = RealNoiseDataset_Byeol(self.config).paths
        # random.shuffle(dataset)

        for i in range(k):
            eighty_percent = int(len(dataset)*0.8)
            nighty_percent = int(len(dataset)*0.9)

            train_partition = dataset[:eighty_percent]
            val_partition = dataset[eighty_percent:nighty_percent]
            test_partition = dataset[nighty_percent:]
            batch_size = self.config['dataset']['batch_size']
            print("training data amount: ", len(train_partition))
            train_loader = DataLoader(CrossValidateDataset(
                train_partition), batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(CrossValidateDataset(
                val_partition), batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(CrossValidateDataset(
                test_partition), batch_size=batch_size, shuffle=True)
            yield train_loader, val_loader, test_loader
            split_point = len(dataset)//k
            dataset = dataset[split_point:]+dataset[:split_point]

    def run(self, k_fold=1):
        loaders_generator = self.partition_dataset(k=k_fold)
        self.k_fold = k_fold
        for k in range(k_fold):
            curr_iter = 0
            # init model
            self.__model = get_model(self.config)
            self.__model = torch.nn.DataParallel(self.__model)
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
            self._val_samples_path = curr_experiment_dir+"/val_sample_imgs"
            os.makedirs(self._val_samples_path, exist_ok=True)
            print(f"cross validation {k+1}/{k_fold}")
            train_loader, val_loader, test_loader = next(loaders_generator)

            self.stop_progressing = 0
            self.__min_val_loss = float("inf")
            for epoch in tqdm(range(self.epoch)):
                # early stop

                if self.stop_progressing >= 10:
                    print("Early stopped :)")
                    break

                # train
                for iter, data in enumerate((train_loader)):
                    noise, ground_truth = data
                    ground_truth, noise = ground_truth, noise
                    ground_truth, noise = ground_truth.to(
                        self.device).float(), noise.to(self.device).float()
                    self.__optimizer.zero_grad()
                    prediction = self.__model.forward(noise)
                    loss = self.__criterion(prediction, ground_truth)
                    self.writer.add_scalar(f'train/loss', loss, curr_iter)
                    loss.backward()
                    orig_SNR, denoised_SNR, SNR_incr = compute_metrics(ground_truth, noise, prediction)
                    self.writer.add_scalar(f'train/orig_SNR', orig_SNR, curr_iter)
                    self.writer.add_scalar(f'train/denoised_SNR', denoised_SNR, curr_iter)
                    self.writer.add_scalar(f'train/SNR_incr', SNR_incr, curr_iter)
                    self.__optimizer.step()
                    curr_iter += 1
                # val
                val_loss = 0
                with torch.no_grad():
                    for iter, data in enumerate((val_loader)):

                        noise, ground_truth = data
                        ground_truth, noise = ground_truth, noise
                        ground_truth, noise = ground_truth.to(self.device).float(), noise.to(self.device).float()
                        prediction = self.__model.forward(noise)
                        val_loss += self.__criterion(prediction, ground_truth)
                    self.writer.add_scalar( f'val/loss', val_loss/(iter+1), epoch)
                    orig_SNR, denoised_SNR, SNR_incr = compute_metrics(ground_truth, noise, prediction)
                    self.writer.add_scalar(f'val/orig_SNR', orig_SNR, curr_iter)
                    self.writer.add_scalar(f'val/denoised_SNR', denoised_SNR, curr_iter)
                    self.writer.add_scalar(f'val/SNR_incr', SNR_incr, curr_iter)
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

                    display_pics(noise[0,0].cpu(), prediction[0,0].cpu(), ground_truth[0,0].cpu(), save_path=self._val_samples_path+f"/epoch_{epoch}")
            # test
            test_loss = 0
            with torch.no_grad():
                for iter, data in enumerate((test_loader)):
                    noise, ground_truth = data
                    ground_truth, noise = ground_truth, noise
                    ground_truth, noise = ground_truth.to(
                        self.device).float(), noise.to(self.device).float()
                    prediction = self.__model.forward(noise)
                    test_loss += self.__criterion(prediction, ground_truth)
                for i in range(len(prediction)):
                    display_pics(noise[0,0].cpu(), prediction[0,0].cpu(), ground_truth[0,0].cpu(), save_path=self._test_samples_path+f"/num_{i}")
                    
                self.writer.add_scalar(f'test/loss', test_loss/(iter+1), 0)
                orig_SNR, denoised_SNR, SNR_incr = compute_metrics(ground_truth, noise, prediction)
                self.writer.add_scalar(f'test/orig_SNR', orig_SNR, curr_iter)
                self.writer.add_scalar(f'test/denoised_SNR', denoised_SNR, curr_iter)
                self.writer.add_scalar(f'test/SNR_incr', SNR_incr, curr_iter)


if len(sys.argv) > 1:
    name = sys.argv[1]
else:
    # default name
    name = "plain_cross_valid"

seed = 3405
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
exp = DenoiseExp(name)
exp.run(k_fold=10)
