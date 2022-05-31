import json
import os
from dataloader import get_datasets
from model_factory import get_model
from weakref import ref
import matplotlib.pyplot as plt
import numpy as np
import torch, copy
from tqdm import tqdm
from datetime import datetime
import shutil

ROOT_STATS_DIR = "./experiment_data"
class Experiment(object):
    def __init__(self, name):
        f = open('./hyperparameters/'+ name + '.json')
        config = data = json.load(f)
        self.config = config

        if config is None:
            raise Exception("Configuration file doesn't exist: ", name)

        self.name=name
        self.__name = config['experiment_name']

        #make directory for this experiement
        self.__experiment_dir = os.path.join(ROOT_STATS_DIR, self.__name)
        if not os.path.exists(self.__experiment_dir):
            os.makedirs(self.__experiment_dir)

        # Load Datasets
        self.__train_loader, self.__val_loader, self.__test_loader = get_datasets(config)

        # Setup Experiment
        self.__epochs = config['experiment']['num_epochs']
        self.__current_epoch = 0
        self.__training_losses = []
        self.__val_losses = []
        self.__min_val_loss = 999999
        self.__learning_rate = config['experiment']['learning_rate']

        # Init Model
        self.__model = get_model(config)
        if config['model']['model_type'] == 'filter':
            return None
        self.best_epoch = 0
        try:
            lr_step = config["experiment"]["lr_scheduler_step"]
        except:
            pass
        try: 
            self.lr_scheduler_type = config["experiment"]["lr_scheduler_type"]
            if self.lr_scheduler_type == "step":
                self.__lr_scheduler = torch.optim.lr_scheduler.StepLR(self.__optimizer, lr_step[0])
            elif self.lr_scheduler_type == "multi_step":
                self.__lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.__optimizer, lr_step)
            elif self.lr_scheduler_type == "criterion":
                self.__lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.__optimizer,'min')
            else: 
                self.__lr_scheduler = None 
        except Exception as e: 
            self.__lr_scheduler = None 

        # Also assign GPU to device
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.__model = self.__model.to(self.device)

        # choosing loss function and optimizer
        if  config["experiment"]["loss_func"] == "MSE":
            self.__criterion = torch.nn.MSELoss()
        else :
            self.__criterion = torch.nn.CrossEntropyLoss() # edited
        
        self.__optimizer = torch.optim.Adam(self.__model.parameters(), lr = self.__learning_rate) # edited

        self.__init_model()

        # Load Experiment Data if available
        # self.__load_experiment()

    def run(self):
        if self.config['model']['model_type'] == 'filter':
            return
        for epoch in range( self.__epochs):  # loop over the dataset multiple times
            train_loss = self.__train()
            val_loss = self.__val()
            self.__record_stats(train_loss, val_loss)
            if self.__lr_scheduler is not None:
                if self.lr_scheduler_type == "criterion":
                    self.__lr_scheduler.step(val_loss)
                else: 
                    self.__lr_scheduler.step()


    def __train(self):
        self.__model.train()
        training_loss = 0
        # temp
        # Iterate over the data, implement the training function
        for iter, data in enumerate(tqdm(self.__train_loader)):
            raw, noise = data
            raw, noise = raw.cuda(), noise.cuda()
           
            self.__optimizer.zero_grad()          
            prediction = self.__model.forward(noise)
            # prediction = prediction.type(torch.float32)
            loss=self.__criterion(prediction,raw )
            loss.backward()
            self.__optimizer.step()
            training_loss+=loss.item()
        training_loss/=(iter+1)
        
        return training_loss

    def __val(self):
        self.__model.eval()
        val_loss = 0
        with torch.no_grad():
            for iter, data in enumerate(tqdm(self.__val_loader)):
                raw, noise = data
                raw, noise = raw.cuda(), noise.cuda()
                prediction = self.__model(noise)
                # prediction = prediction.type(torch.float32)
                loss=self.__criterion(prediction,raw )
                val_loss+=loss.item()            
        val_loss = val_loss/(iter+1)
        if val_loss < self.__min_val_loss:
            self.__min_val_loss = val_loss
            self.__save_model()
            self.best_epoch = self.__current_epoch
        output_msg = "Current validation loss: " + str(val_loss)
        # send_discord_msg(output_msg)
        return val_loss

    def test(self):
        accu = []
        displayed = False
        if self.config['model']['model_type'] == 'filter':
            for iter, data in enumerate(tqdm(self.__test_loader)):
                raw, noise = data
                if not displayed:
                    ax = plt.subplot(1, 4, 1)
                    plt.tight_layout()
                    ax.set_title('orig')
                    ax.axis('off')
                    plt.imshow(raw[0])
                    displayed = True
                perdiction = self.__model(noise)
                
                # print (torch.sum(raw))
                # print (np.sum(np.array(raw) * perdiction ))
                batch_accu = np.sum(np.array(raw) * perdiction )/(torch.sum(raw))
                accu.append(batch_accu)
                print(accu)
            print(sum(accu) / len(accu))
            return sum(accu) / len(accu)

    def __init_model(self):
        if torch.cuda.is_available():
            self.__model = self.__model.cuda().float()
            self.__criterion = self.__criterion.cuda()

    def __record_stats(self, train_loss, val_loss):
        self.__training_losses.append(train_loss)
        self.__val_losses.append(val_loss)

        self.plot_stats()

        write_to_file_in_dir(self.__experiment_dir, 'training_losses.txt', self.__training_losses)
        write_to_file_in_dir(self.__experiment_dir, 'val_losses.txt', self.__val_losses)

    def __save_model(self):
        root_model_path = os.path.join(self.__experiment_dir, 'latest_model.pt')
        model_dict = self.__model.state_dict()
        state_dict = {'model': model_dict, 'optimizer': self.__optimizer.state_dict()}
        torch.save(state_dict, root_model_path)

def write_to_file_in_dir(root_dir, file_name, data):
    path = os.path.join(root_dir, file_name)
    write_to_file(path, data)

def write_to_file(path, data):
    with open(path, "w") as outfile:
        json.dump(data, outfile)