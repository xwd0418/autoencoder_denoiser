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
        config = json.load(f)
        config['experiment_name'] = name
        self.config = config
        self.best_model = None

        if config is None:
            raise Exception("Configuration file doesn't exist: ", name)

        self.name=name
        self.__name = config['experiment_name']

        # make directory for this experiement
        self.__experiment_dir = os.path.join(ROOT_STATS_DIR, self.__name)

        # Load Datasets
        self.__train_loader, self.__val_loader, self.__test_loader = get_datasets(config)

        # Setup Experiment
        self.__epochs = config['experiment']['num_epochs']
        self.__curr_epoch = 0
        self.__current_epoch = 0
        self.__training_losses = []
        self.__val_losses = []
        self.__training_accu = []
        self.__val_accu = []
        self.__min_val_loss = 999999
        self.__learning_rate = config['experiment']['learning_rate']

        # Init Model
        self.__model = get_model(config)
        # self.__model = torch.nn.DataParallel(self.__model)
        
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
        self.__load_experiment()

    # Loads the experiment data if exists to resume training from last saved checkpoint.
    def __load_experiment(self):
        os.makedirs(ROOT_STATS_DIR, exist_ok=True)

        if os.path.exists(self.__experiment_dir):
            try:
                training_losses = read_file_in_dir(self.__experiment_dir, 'training_losses.txt')
                val_losses = read_file_in_dir(self.__experiment_dir, 'val_losses.txt')
                self.__training_accu = read_file_in_dir(self.__experiment_dir, 'training_accu.txt')
                self.__val_accu = read_file_in_dir(self.__experiment_dir, 'val_accu.txt')

                current_epoch = len(self.__training_accu)
 
                state_dict = torch.load(os.path.join(self.__experiment_dir, 'latest_model.pt'))

                self.__model.load_state_dict(state_dict['model'])
                self.__optimizer.load_state_dict(state_dict['optimizer'])

                self.__training_losses = training_losses
                self.__val_losses = val_losses
                self.__current_epoch = current_epoch
                print("Successfully loaded previous model")
            except NotRegularFileError:
                print('Reading last expriment failed, removing folder')
                shutil.rmtree(self.__experiment_dir)
                os.makedirs(self.__experiment_dir)
        else:
            os.makedirs(self.__experiment_dir)

    def run(self):
        if self.config['model']['model_type'] == 'filter':
            return
        for i in range( self.__epochs):  # loop over the dataset multiple times
            self.__current_epoch+=1
            print("epoch: ",self.__current_epoch)
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
            raw, noise = self.__move_to_cuda(data)
           
            self.__optimizer.zero_grad()
            # print ("noise shape",noise.shape)        
            
            prediction = self.__model.forward(noise)
            # prediction = prediction.type(torch.float32)
            # print(prediction.shape, raw.shape)
            ground_truth = raw
            if self.config["experiment"]["loss_func"] == "CrossEntropy":
                ground_truth = raw[:,0,:,:]
            loss=self.__criterion(prediction,ground_truth )
            prediction = torch.clip(prediction.round(),0,1)
            with torch.no_grad():
                intersec = np.sum(np.array(raw.cpu()) * np.array(prediction.cpu()))
                union = torch.sum(raw)+torch.sum(prediction)-intersec
                accu =intersec / union
                if (accu > 1 or accu < 0) :
                    print("bug!")
                    print("raw is ", torch.sum(raw))
                    print("predict is ",torch.sum(prediction) )
                    print("interesct is ", intersec )
                    print("union is " , union)
                    
            loss.backward()
            self.__optimizer.step()
            training_loss+=loss.item()
            training_accu+=accu.item()
        training_loss/=(iter+1)
        training_accu/=(iter+1)
        
        
        return training_loss,training_accu

    

    def __val(self):
        # print("validating stage")

        accu = []
        self.__model.eval()
        val_loss = 0
        with torch.no_grad():
            for iter, data in enumerate(tqdm(self.__val_loader)):
                raw, noise = self.__move_to_cuda(data)
                prediction = self.__model.forward(noise)
                
                # find loss
                # prediction = prediction.type(torch.float32)
                ground_truth = raw
                if self.config["experiment"]["loss_func"] == "CrossEntropy":
                    ground_truth = raw[:,0,:,:]
                loss=self.__criterion(prediction,ground_truth )
                val_loss+=loss.item() 
                
                #make prediction to int values
                prediction = torch.clip(prediction.round(),0,1)
                
                #draw sample pics
                if self.__current_epoch% 15 ==0 and iter==0:
                    
                    if self.config['model']['model_type'] != 'filter' and self.config['model']['model_type'] != 'vanilla':
                        noise_pic , prediction_pic, raw_pic = noise[0],prediction[0], raw[0]
                    else: noise_pic , prediction_pic, raw_pic = noise,prediction, raw
                    plt.clf()

                    ax = plt.subplot(1, 3, 1)
                    plt.tight_layout()
                    ax.set_title('orig')
                    ax.axis('off')
                    plt.imshow(raw_pic[0].cpu(),cmap='gray')

                    ax = plt.subplot(1, 3, 2)
                    plt.tight_layout()
                    ax.set_title('noise')
                    ax.axis('off')
                    plt.imshow(noise_pic[0].cpu(),cmap='gray')

                    ax = plt.subplot(1, 3, 3)
                    plt.tight_layout()
                    ax.set_title('predicted')
                    ax.axis('off')
                    plt.imshow(prediction_pic[0].cpu(),cmap='gray')

                    plt.savefig(os.path.join(self.__experiment_dir, "epoch_{}_sample_images.png".format(str(self    .__current_epoch))))
                    displayed = True
                    plt.clf()

                # find accuracy
                intersec = np.sum(np.array(raw.cpu()) * np.array(prediction.cpu() ))
                batch_accu =intersec /(torch.sum(raw)+torch.sum(prediction)-intersec)                   
                accu.append(batch_accu)

           
        val_loss = val_loss/(iter+1)
        
        print("val accuracy", (sum(accu) / len(accu)).item()  )  
        print("loss: ", val_loss) 

        if val_loss < self.__min_val_loss:
            self.__min_val_loss = val_loss
            self.__save_model()
            print("best model updated")
            self.best_epoch = self.__current_epoch
            self.best_model = copy.deepcopy(self.__model)

        return val_loss,(sum(accu) / len(accu)).item()  

    def test(self):
        print("testing stage")
        accu = []
        test_loss= 0
        displayed = False

        """if filter """
        if self.config['model']['model_type'] == 'filter':
            for iter, data in enumerate(tqdm(self.__test_loader)):
                raw, noise = data
                prediction = self.__model(noise)
                
                prediction = torch.clip(prediction.round(),0,1)
                
                if not displayed:
                    noise_pic , prediction_pic, raw_pic = noise,prediction, raw
                    plt.clf()

                    ax = plt.subplot(1, 3, 1)
                    plt.tight_layout()
                    ax.set_title('orig')
                    ax.axis('off')
                    plt.imshow(raw_pic[0],cmap='gray')

                    ax = plt.subplot(1, 3, 2)
                    plt.tight_layout()
                    ax.set_title('noise')
                    ax.axis('off')
                    plt.imshow(noise_pic[0],cmap='gray')

                    ax = plt.subplot(1, 3, 3)
                    plt.tight_layout()
                    ax.set_title('predicted')
                    ax.axis('off')
                    plt.imshow(prediction_pic[0],cmap='gray')

                    plt.savefig(os.path.join(self.__experiment_dir, "sample_images.png"))
                    displayed = True
                    plt.clf()
                    
                
                
                # print (torch.sum(raw))
                # print (np.sum(np.array(raw) * perdiction )) 
                intersec = np.sum(np.array(raw) * prediction )
                batch_accu = intersec/(torch.sum(raw)+np.sum(prediction)-intersec)
                # batch_accu = np.sum(np.array(raw) * np.array(prediction ))/(np.sum(raw)+np.sum(prediction))
                accu.append(batch_accu)
                # print(accu)
            print("avg testing accuracy is ",(sum(accu) / len(accu)))
            
            return sum(accu) / len(accu)

        """if auto encoder i.e. not using filter"""
        with torch.no_grad():
            for iter, data in enumerate(tqdm(self.__test_loader)):
                raw, noise = self.__move_to_cuda(data)
                prediction = self.best_model(noise).data
                prediction = torch.clip(prediction.round(),0,1)
                
                ground_truth = raw
                if self.config["experiment"]["loss_func"] == "CrossEntropy":
                    ground_truth = raw[:,0,:,:]
                loss=self.__criterion(prediction,ground_truth )
                test_loss+=loss.item() 
                
                if not displayed:
                    if self.config['model']['model_type'] != 'filter' and self.config['model']['model_type'] != 'vanilla':
                        noise_pic , prediction_pic, raw_pic = noise[0],prediction[0], raw[0]
                    else: noise_pic , prediction_pic, raw_pic = noise,prediction, raw
                    
                    if self.config["model"]['model_type'] == "JNet":
                        noise_pic = noise_pic[0]
                    
                    plt.clf()

                    ax = plt.subplot(1, 3, 1)
                    plt.tight_layout()
                    ax.set_title('orig')
                    ax.axis('off')
                    plt.imshow(raw_pic[0].cpu(),cmap='gray')

                    ax = plt.subplot(1, 3, 2)
                    plt.tight_layout()
                    ax.set_title('noise')
                    ax.axis('off')
                    plt.imshow(noise_pic[0].cpu(),cmap='gray')

                    ax = plt.subplot(1, 3, 3)
                    plt.tight_layout()
                    ax.set_title('predicted')
                    ax.axis('off')
                    plt.imshow(prediction_pic[0].cpu(),cmap='gray')

                    plt.savefig(os.path.join(self.__experiment_dir, "sample_images.png"))
                    displayed = True
                    plt.clf()
                
                
                # print (torch.sum(raw))
                # print (np.sum(np.array(raw) * perdiction ))
                intersec = np.sum(np.array(raw.cpu()) * np.array(prediction.cpu() ))
                intersec = np.sum(np.array(raw.cpu()) * np.array(prediction.cpu() ))
                union = (torch.sum(raw)+torch.sum(prediction)-intersec)
                batch_accu =intersec / union
                accu.append(batch_accu)
                if (batch_accu > 1 or batch_accu < 0) :
                    print("bug!")
                    print("raw is ", torch.sum(raw))
                    print("predict is ",torch.sum(prediction) )
                    print("interesct is ", intersec )
                    print("union is " , union)
                # print(accu)
                
            test_loss /= len(accu)
            avg_accu = (sum(accu) / len(accu)).item()
            print("avg testing accuracy is ",avg_accu)
            print("avg testing loss is ", test_loss)
            
            output_msg = {"loss":test_loss, "accuracy":avg_accu}
            write_to_file_in_dir(self.__experiment_dir, 'testing result.txt', output_msg)
            return avg_accu

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
        write_to_file_in_dir(self.__experiment_dir, 'training_accu.txt', self.__training_accu)
        write_to_file_in_dir(self.__experiment_dir, 'val_accu.txt', self.__val_accu)

    def __save_model(self):
        root_model_path = os.path.join(self.__experiment_dir, 'latest_model.pt')
        model_dict = self.__model.state_dict()
        state_dict = {'model': model_dict, 'optimizer': self.__optimizer.state_dict()}
        torch.save(state_dict, root_model_path)

    def __move_to_cuda(self, data):
        raw, noise = data
        if type(noise) is list:
            raw= raw.cuda().float()
            noisy_sample, tessellated_noise = noise
            noise = noisy_sample.cuda().float(),  tessellated_noise.cuda().float()
        else:
            
            raw, noise = raw.cuda().float(), noise.cuda().float()
        return raw,noise
    
    def plot_stats(self):
        #training
        e = len(self.__training_losses)
        x_axis = np.arange(1, e + 1, 1)
        plt.figure()
        plt.plot(x_axis, self.__training_losses, label="Training Loss")
        plt.plot(x_axis, self.__val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.legend(loc='best')
        plt.title(self.__name + " Loss Plot")
        plt.savefig(os.path.join(self.__experiment_dir, "loss_plot.png"))
        
        #validation        
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
        
        
        if len(self.__training_losses) >5 :
            #training without first few
            plt.clf()
            e = len(self.__training_losses) -5
            x_axis = np.arange(1, e + 1, 1)
            plt.figure()
            plt.plot(x_axis, self.__training_losses[5:], label="Training Loss")
            plt.plot(x_axis, self.__val_losses[5:], label="Validation Loss")
            plt.xlabel("Epochs")
            plt.legend(loc='best')
            plt.title(self.__name + " Loss Plot")
            plt.savefig(os.path.join(self.__experiment_dir, "loss_plot_without_heads.png"))
            
            #training without first few
            plt.clf()
            e = len(self.__training_losses) -5
            x_axis = np.arange(1, e + 1, 1)
            plt.figure()
            plt.plot(x_axis, self.__training_accu[5:], label="Training Accu")
            plt.plot(x_axis, self.__val_accu[5:], label="Validation Accu")
            plt.xlabel("Epochs")
            plt.legend(loc='best')
            plt.title(self.__name + " Accu Plot")
            plt.savefig(os.path.join(self.__experiment_dir, "accu_plot_without_heads.png"))
            
            

def write_to_file_in_dir(root_dir, file_name, data):
    path = os.path.join(root_dir, file_name)
    write_to_file(path, data)

def write_to_file(path, data):
    with open(path, "w") as outfile:
        json.dump(data, outfile, indent=2)
   
class NotRegularFileError(Exception):
    pass
     
def read_file_in_dir(root_dir, file_name):
    path = os.path.join(root_dir, file_name)
    return read_file(path)

def read_file(path):
    if os.path.isfile(path):
        with open(path) as json_file:
            data = json.load(json_file)
        return data
    else:
        raise NotRegularFileError("not an existing regular file: ", path)