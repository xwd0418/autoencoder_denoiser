import json, sys, os, pickle, random
import numpy as np
import torch, copy, cv2
from tqdm import tqdm
from model_factory import get_model , CDANLoss
import matplotlib.image
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import LearningRateMonitor

class CrossDataset(Dataset):
    def __init__(self, partition) -> None:
        super().__init__() 
        self.imgs = partition

    def  __len__(self):
        return len(self.imgs)
        
    def __getitem__(self, index):
        return self.imgs[index]

class TransferDenoiseExp(object):
    def __init__(self, name) -> None:
        config_dir  = '/root/autoencoder_denoiser/configs_transfer_learning'  
        f = open(f'{config_dir}/'+ name + '.json')
        config = json.load(f)
        self.config = config
        #init dirs and plotting tools
        self.config = config
        self.__experiment_dir = '/root/autoencoder_denoiser/exps/transfer_learning/'+name
        
        clist = [(0,"green"), (0.5,"white"), (1, "red")]
        self.custom_diff_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("_",clist)
        clist = [(0,"darkblue"), (0.5,"white"), (1, "darkred")]
        self.custom_HSQC_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("_",clist)
        
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
            
    def plot(self, ground_truth, noise, prediction, path, batch_index = None):
        prediction = torch.clip(prediction,0,1)
        if batch_index == None:
            batch_index = random.randrange(8)
        plt.clf()

        ax = plt.subplot(2, 2, 1)
        plt.tight_layout()
        ax.set_title('original')
        ax.axis('off')
        plt.imshow(ground_truth[batch_index].view((180,120)).cpu(),cmap=self.custom_HSQC_cmap, vmax=1, vmin=-1)

        ax = plt.subplot(2, 2, 2)
        plt.tight_layout()
        ax.set_title('noise')
        ax.axis('off')
        plt.imshow(noise[batch_index].view((180,120)).cpu(),cmap=self.custom_HSQC_cmap, vmax=1, vmin=-1)

        ax = plt.subplot(2, 2, 3)
        plt.tight_layout()
        ax.set_title('predicted')
        ax.axis('off')
        plt.imshow(prediction[batch_index].view((180,120)).cpu(),cmap=self.custom_HSQC_cmap, vmax=1, vmin=-1)
                
        ax = plt.subplot(2, 2, 4)
        plt.tight_layout()
        ax.set_title('difference')
        ax.axis('off')
                
                # difference = prediction_pic[0].cpu()-raw_pic[0].cpu()
                # difference = difference.float()/2 + 0.5
                # print(difference)
        difference = cv2.subtract(np.array(prediction[batch_index].view((180,120)).cpu()), np.array(ground_truth[batch_index].view((180,120)).cpu()))
        plt.imshow(difference, cmap = self.custom_diff_cmap, vmax=1, vmin=-1)

                # print(os.path.join(self._test_samples_path, f"sample_image{displayed}.png"))
        plt.savefig(path)
    
    def partition_dataset(self,k):
        with open('/root/autoencoder_denoiser/dataset/imgs_as_array.pkl', 'rb') as f:
            imgs =  pickle.load(f)
            random.shuffle(imgs)

            for i in range(k):
                eighty_percent = int(len(imgs)*0.8)
                nighty_percent = int(len(imgs)*0.9)
                
                train_partition = imgs[:eighty_percent]
                val_partition = imgs[eighty_percent:nighty_percent]
                test_partition = imgs[nighty_percent:]
                batch_size = self.config['experiment']['batch_size']
                print("training data amount: ", len(train_partition))
                train_loader = DataLoader(CrossDataset(train_partition), batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(CrossDataset(val_partition), batch_size=batch_size,shuffle=True)
                test_loader = DataLoader(CrossDataset(test_partition), batch_size=batch_size,shuffle=True)
                yield train_loader, val_loader, test_loader
                split_point = len(imgs)//k
                imgs = imgs[split_point:]+imgs[:split_point]
                
    
    def run(self, k_fold=1):
        loaders_generator = self.partition_dataset(k=k_fold)
        self.k_fold = k_fold    
        for k in range(k_fold):
            # init model
            self.__model = get_model(self.config)
            self.__model = torch.nn.DataParallel(self.__model)
            if torch.cuda.is_available():
                self.__model = self.__model.cuda().float()
            self.load_model(self.config['loading_path'])
            
            # init training configs 
            
            self.__learning_rate = self.config['experiment']['learning_rate']
            self.__optimizer = torch.optim.Adam(self.__model.parameters(), lr = self.__learning_rate) # edited
            self.__lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.__optimizer,'min')
        
        
            curr_experiment_dir = self.__experiment_dir+f'/fold_{k+1}'
            self.writer = SummaryWriter(log_dir=curr_experiment_dir)
            self._test_samples_path = curr_experiment_dir+"/testing_sample_imgs"
            os.makedirs(self._test_samples_path, exist_ok=True)
            self._val_samples_path = curr_experiment_dir+"/val_sample_imgs"
            os.makedirs(self._val_samples_path, exist_ok=True)
            print(f"cross validation {k+1}/{k_fold}") 
            train_loader, val_loader, test_loader = next(loaders_generator)
           
            curr_iter = 0
            self.stop_progressing=0
            self.__min_val_loss = float("inf")
            for epoch in tqdm(range(self.epoch)):
                # early stop
                
                if self.stop_progressing >= 35:
                    print("Early stopped :)")
                    break
                
                #train
                for iter, data in enumerate((train_loader)):
                    noise, ground_truth = data   
                    ground_truth, noise = ground_truth.unsqueeze(1), noise.unsqueeze(1)
                    ground_truth, noise = ground_truth.to(self.device).float(), noise.to(self.device).float()
                    self.__optimizer.zero_grad()
                    prediction = self.__model.forward(noise)
                    loss = self.__criterion(prediction,ground_truth)
                    self.writer.add_scalar(f'train/loss', loss, curr_iter)        
                    loss.backward()
                    self.__optimizer.step()
                    curr_iter+=1
                #val
                val_loss = 0
                with torch.no_grad(): 
                    for iter, data in enumerate((val_loader)):
                     
                        noise, ground_truth = data   
                        ground_truth, noise = ground_truth.unsqueeze(1), noise.unsqueeze(1)
                        ground_truth, noise = ground_truth.to(self.device).float(), noise.to(self.device).float()
                        prediction = self.__model.forward(noise)
                        val_loss += self.__criterion(prediction,ground_truth)
                    self.writer.add_scalar(f'val/loss', val_loss/(iter+1), epoch) 
                    self.__lr_scheduler.step(val_loss)
                #update early stop
                if val_loss < self.__min_val_loss:
                    self.__min_val_loss = val_loss
                    torch.save(self.__model, curr_experiment_dir+"/best_model.pt")
                    # print("best model updated")
                    self.best_model = copy.deepcopy(self.__model)
                    self.stop_progressing = 0
                else:
                    self.stop_progressing += 1
                
                if epoch % 20==0:
                    self.plot(ground_truth, noise, prediction, path = self._val_samples_path+f"/epoch_{epoch}")
            #test        
            test_loss = 0
            with torch.no_grad(): 
                for iter, data in enumerate((test_loader)):
                    noise, ground_truth = data   
                    ground_truth, noise = ground_truth.unsqueeze(1), noise.unsqueeze(1)
                    ground_truth, noise = ground_truth.to(self.device).float(), noise.to(self.device).float()
                    prediction = self.__model.forward(noise)
                    test_loss += self.__criterion(prediction,ground_truth)
                for i in range(len(prediction)):
                    self.plot(ground_truth, noise, prediction, path = self._test_samples_path+f"/num_{i}", batch_index=i)     
                self.writer.add_scalar(f'test/loss', test_loss/(iter+1), 0)    
        
                     
    
    
if len(sys.argv) > 1:
        name = sys.argv[1]
else: 
        raise Exception("which config to run?")

seed = 114414
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
exp = TransferDenoiseExp(name)
exp.run()
