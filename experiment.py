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
        self.__experiment_dir = os.path.join(ROOT_STATS_DIR, self.__name)

        # Load Datasets
        self.__train_loader, self.__val_loader, self.__test_loader = get_datasets(config)

        # Setup Experiment
        self.__epochs = config['experiment']['num_epochs']
        self.__current_epoch = 0
        self.__training_losses = []
        self.__val_losses = []
        self.__min_val_loss = 999999
        self.__best_model = None  # Save your best model in this field and use this in test method.
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

        # TODO: Set these Criterion and Optimizers Correctly
        # Also assign GPU to device
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.__model = self.__model.to(self.device)
        self.__criterion = torch.nn.CrossEntropyLoss(ignore_index=0) # edited
        self.__optimizer = torch.optim.Adam(self.__model.parameters(), lr = self.__learning_rate) # edited

        self.__init_model()

        # Load Experiment Data if available
        self.__load_experiment()

    def run(self):
        if self.config['model']['model_type'] == 'filter':
            return

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

