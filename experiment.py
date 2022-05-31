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
        self.__training_accu = []
        self.__val_accu = []
        self.__min_val_loss = 999999
        self.__learning_rate = config['experiment']['learning_rate']

        # Init Model
        self.__model = get_model(config)
        
        if config['model']['model_type'] == 'filter':
            return None
        self.best_epoch = 0

        # Also assign GPU to device
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        
        print("model got")
        self.__model = self.__model.to(self.device)
        print("model finish moving")

        print(" choosing loss function and optimizer")
        if  config["experiment"]["loss_func"] == "MSE":
            self.__criterion = torch.nn.MSELoss()
        else :
            self.__criterion = torch.nn.CrossEntropyLoss() # edited
        
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
        # self.__load_experiment()

    def run(self):
        if self.config['model']['model_type'] == 'filter':
            return
        for epoch in range( self.__epochs):  # loop over the dataset multiple times
            print("epoch: ",epoch)
            train_loss, train_accu = self.__train()
            val_loss,val_accu = self.__val()
            
            if self.__lr_scheduler is not None:
                if self.lr_scheduler_type == "criterion":
                    self.__lr_scheduler.step(val_loss)
                else: 
                    self.__lr_scheduler.step()
            self.__record_stats(train_loss,train_accu, val_loss,val_accu)
        self.plot_stats()


    def __train(self):
        self.__model.train()
        training_loss = 0
        training_accu = 0
        # temp
        # Iterate over the data, implement the training function
        for iter, data in enumerate(tqdm(self.__train_loader)):
            raw, noise = data
            raw, noise = raw.cuda().float(), noise.cuda().float()
           
            self.__optimizer.zero_grad()          
            prediction = self.__model.forward(noise)
            # prediction = prediction.type(torch.float32)
            # print(prediction.shape, raw.shape)
            loss=self.__criterion(prediction,raw )
            with torch.no_grad():
                accu = np.sum(np.array(raw.cpu()) * np.array(prediction.cpu() ))/(torch.sum(raw))
            loss.backward()
            self.__optimizer.step()
            training_loss+=loss.item()
            training_accu+=accu.item()
        training_loss/=(iter+1)
        training_accu/=(iter+1)
        
        
        return training_loss,training_accu

    def __val(self):
        print("validating stage")

        accu = []
        self.__model.eval()
        val_loss = 0
        with torch.no_grad():
            for iter, data in enumerate(tqdm(self.__val_loader)):
                raw, noise = data
                raw, noise = raw.cuda().float(), noise.cuda().float()
                prediction = self.__model.forward(noise)
                
                # find accuracy
                with torch.no_grad():
                    batch_accu = np.sum(np.array(raw.cpu()) * np.array(prediction.cpu() ))/(torch.sum(raw))
                    accu.append(batch_accu)

                # find loss
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

           
        return val_loss,(sum(accu) / len(accu)).item()  

    def test(self):
        print("testing stage")
        accu = []
        displayed = False

        """if filter """
        if self.config['model']['model_type'] == 'filter':
            for iter, data in enumerate(tqdm(self.__test_loader)):
                raw, noise = data
                raw, noise = raw.cuda().float(), noise.cuda().float()
                # if not displayed:
                #     ax = plt.subplot(1, 4, 1)
                #     plt.tight_layout()
                #     ax.set_title('orig')
                #     ax.axis('off')
                #     plt.imshow(raw[0])
                #     displayed = True
                prediction = self.__model(noise)
                
                # print (torch.sum(raw))
                # print (np.sum(np.array(raw) * perdiction ))
                batch_accu = np.sum(np.array(raw.cpu()) * np.array(prediction.cpu() ))/(torch.sum(raw))
                accu.append(batch_accu)
                # print(accu)
            print((sum(accu) / len(accu)))
            return sum(accu) / len(accu)

        """if auto encoder"""
        with torch.no_grad():
            for iter, data in enumerate(tqdm(self.__test_loader)):
                raw, noise = data
                raw, noise = raw.cuda().float(), noise.cuda().float()
                prediction = self.__model(noise).data

                if not displayed:
                    plt.clf()

                    ax = plt.subplot(1, 3, 1)
                    plt.tight_layout()
                    ax.set_title('orig')
                    ax.axis('off')
                    plt.imshow(raw[0].cpu())

                    ax = plt.subplot(1, 3, 2)
                    plt.tight_layout()
                    ax.set_title('noise')
                    ax.axis('off')
                    plt.imshow(noise[0].cpu())

                    ax = plt.subplot(1, 3, 3)
                    plt.tight_layout()
                    ax.set_title('predicted')
                    ax.axis('off')
                    plt.imshow(prediction[0].cpu())

                    plt.savefig(os.path.join(self.__experiment_dir, "sample_images.png"))
                    displayed = True
                    plt.clf()
                
                
                # print (torch.sum(raw))
                # print (np.sum(np.array(raw) * perdiction ))
                batch_accu = np.sum(np.array(raw.cpu()) * np.array(prediction.cpu() ))/(torch.sum(raw))
                accu.append(batch_accu)
                # print(accu)
            print((sum(accu) / len(accu)))
            return sum(accu) / len(accu)

    def __init_model(self):
        if torch.cuda.is_available():
            self.__model = self.__model.cuda().float()
            self.__criterion = self.__criterion.cuda()

    def __record_stats(self, train_loss,train_accu, val_loss,val_accu):
        self.__training_losses.append(train_loss)
        self.__val_losses.append(val_loss)
        self.__training_accu.append(train_accu)
        self.__val_accu.append(val_accu)

        write_to_file_in_dir(self.__experiment_dir, 'training_losses.txt', self.__training_losses)
        write_to_file_in_dir(self.__experiment_dir, 'val_losses.txt', self.__val_losses)

    def __save_model(self):
        root_model_path = os.path.join(self.__experiment_dir, 'latest_model.pt')
        model_dict = self.__model.state_dict()
        state_dict = {'model': model_dict, 'optimizer': self.__optimizer.state_dict()}
        torch.save(state_dict, root_model_path)

    def plot_stats(self):
        e = len(self.__training_losses)
        x_axis = np.arange(1, e + 1, 1)
        plt.figure()
        plt.plot(x_axis, self.__training_losses, label="Training Loss")
        plt.plot(x_axis, self.__val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.legend(loc='best')
        plt.title(self.__name + " Loss Plot")
        plt.savefig(os.path.join(self.__experiment_dir, "loss_plot.png"))
        # plt.show()
        
        plt.clf()
        e = len(self.__training_accu)
        x_axis = np.arange(1, e + 1, 1)
        plt.figure()
        plt.plot(x_axis, self.__training_accu, label="Training Accuracy")
        plt.plot(x_axis, self.__val_accu, label="Validation Accuracy")
        plt.xlabel("Epochs")
        plt.legend(loc='best')
        plt.title(self.__name + " Accu Plot")
        plt.savefig(os.path.join(self.__experiment_dir, "accu_plot.png"))

def write_to_file_in_dir(root_dir, file_name, data):
    path = os.path.join(root_dir, file_name)
    write_to_file(path, data)

def write_to_file(path, data):
    with open(path, "w") as outfile:
        json.dump(data, outfile)