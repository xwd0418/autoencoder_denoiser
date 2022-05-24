import json
import os
from dataloader import get_datasets

ROOT_STATS_DIR = "./experiment_data"
class Experiment(object):
    def __init__(self, name):
        f = open('./hyperparameters/', name + '.json')
        config_data = data = json.load(f)
        
        if config_data is None:
            raise Exception("Configuration file doesn't exist: ", name)

        self.name=name
        self.__name = config_data['experiment_name']
        self.__experiment_dir = os.path.join(ROOT_STATS_DIR, self.__name)

        # Load Datasets
        self.__train_loader, self.__val_loader, self.__test_loader = get_datasets(config_data)

        # Setup Experiment
        self.__epochs = config_data['experiment']['num_epochs']
        self.__current_epoch = 0
        self.__training_losses = []
        self.__val_losses = []
        self.__min_val_loss = 999999
        self.__best_model = None  # Save your best model in this field and use this in test method.
        self.__learning_rate = config_data['experiment']['learning_rate']
        # Init Model
        self.__model = get_model(config_data, self.__vocab)
        self.best_epoch = 0
        try:
            lr_step = config_data["experiment"]["lr_scheduler_step"]
        except:
            pass
        try: 
            self.lr_scheduler_type = config_data["experiment"]["lr_scheduler_type"]
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