import json
import os
import matplotlib.pyplot as plt
from torchmetrics.functional import precision_recall
"""
This file is used to test a model across all real hsqc images
    
In other word, if a method uses real_images during supervised training, this file should not be used during testing. 
Instead, it should use cross validation 
"""

import numpy as np
import json
from glob import glob
import torch, sys
from model_factory import get_model
from model_factory import UNet
from hsqc_dataset import *
from tqdm import tqdm
from utils import display_pics, compute_metrics

"""configurations"""
device = torch.device("cuda:0")
EXP_VERSION = "BCE"
CONFIG_NAME = "pos_weight_500"
config_path = f"/root/autoencoder_denoiser/configs_{EXP_VERSION}"
exp_dir = f'/root/autoencoder_denoiser/exps/results_{EXP_VERSION}'

class Test():
    '''
    a Test object, 
    mainly to bind with a config json file and load model weights  
    '''
    def __init__(self, model_name = CONFIG_NAME, 
                    config_path = config_path,
                    exp_dir = exp_dir,
                    threshold=None) -> None:
        
        # currently not use : dilation and resize
        
        self.dilation = False
        self.resize = True
        self.config = None
        name = model_name
        self.name = name

        f = open(f'{config_path}/'+ name + '.json')
        self.config = json.load(f)
        if threshold :
            return
        experiment_path = f"{exp_dir}/{name}/" 
        print('load from: ', os.path.join(experiment_path, 'latest_model.pt'))
        state_dict = torch.load(os.path.join(experiment_path, 'latest_model.pt'))
        model = get_model(self.config)
        # try:
        model.load_state_dict(state_dict['model'])
        model = torch.nn.DataParallel(model)
        # except:
        #     model = torch.nn.DataParallel(model)
        #     model.load_state_dict(state_dict['model'])
            
        model.to(device)
        model.eval()
        self.model = model

        





# dann_test_loader = DataLoader(RealNoiseDataset_Byeol(dann_test.config,range_low=0,range_high=2, show_name=True),
                        #  batch_size=1, shuffle=False, num_workers=8)



def test(model_test: Test):
    config = model_test.config
    test_loader = DataLoader(RealNoiseDataset_Byeol(config,range_low=0,range_high=100, show_name=True),
                        batch_size=1, shuffle=False, num_workers=8)
    
    need_display = True
    testing_result_dir = f'/root/autoencoder_denoiser/testing_real_imgs_results/{model_test.name}/'
    os.makedirs(testing_result_dir, exist_ok= True)   
    displayed=0
    if  config["experiment"]["loss_func"] == "MSE":
        criterion = torch.nn.MSELoss()
    elif config["experiment"]["loss_func"] == "CrossEntropy":
        criterion = torch.nn.CrossEntropyLoss() # edited
    elif config["experiment"]["loss_func"] == "BCE":
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([config['experiment']['BCE_pos_weight']]).to(device)) # edited
    else:
        raise Exception("what is your loss function??")
    loss = 0
    orig_SNR, denoised_SNR, SNR_incr = 0,0,0
    precision, recall = 0, 0
    plt.rcParams["figure.figsize"] = (20,10)
    with torch.no_grad():
        for iter, data in enumerate(tqdm(test_loader)):
            # print(data)
            noise, raw, name = data
            if len(raw.shape)==3:   
                raw, noise = raw.unsqueeze(1), noise.unsqueeze(1)
            raw, noise = raw.to(device).float(), noise.to(device).float()
            prediction = model_test.model.forward(noise)
            
            # find loss
            # prediction = prediction.type(torch.float32)
            ground_truth = raw
        
            # add adv loss !!!
            prediction = torch.clip(prediction,-1,1)

        # print(denoised_1.shape)
        # print(ground_truth.shape)
            loss += criterion(prediction,ground_truth )
            # updating orig_SNR, denoised_SNR, SNR_incr
            orig_SNR, denoised_SNR, SNR_incr = [sum(x) for x in zip( (orig_SNR, denoised_SNR, SNR_incr) , \
                                                                        compute_metrics(raw, noise, prediction))] 
            if config["experiment"]["loss_func"] == "BCE":
                prediction = torch.where(prediction > 0, 1.0, 0.0)
            else:
                prediction = torch.where(torch.abs(prediction) > 0.02, 1.0, 0.0)
            ground_truth = torch.where(ground_truth != 0, 1.0, 0.0)
            precision_cur ,recall_cur = precision_recall(prediction,ground_truth.int())
            precision += precision_cur
            recall += recall_cur
            # f1_score = 2*precision*recall/(precision+recall)
                        
                   
            if need_display and displayed<10:
                noise_pic , prediction_pic, raw_pic = noise[0],prediction[0], raw[0]          
                save_path = testing_result_dir+f"{iter}.png"
                print(save_path)
                display_pics(noise_pic[0].cpu(), prediction_pic[0].cpu(), raw_pic[0].cpu(),save_path=save_path)
                displayed = displayed+1
        
                
        loss /= len(test_loader.dataset)  
        orig_SNR /=   len(test_loader.dataset)
        denoised_SNR /=   len(test_loader.dataset)
        SNR_incr /=   len(test_loader.dataset)
        precision /=   len(test_loader.dataset)
        recall /=   len(test_loader.dataset)
        
        doc_string = f"test loader size:{len(test_loader.dataset)}\n" \
                +f"loss of model: {model_test.name} is {loss}\n" \
               + f"orig_SNR, denoised_SNR, SNR_incr of model: {model_test.name} is {orig_SNR, denoised_SNR, SNR_incr}\n" \
                + f"precision: {precision}, recall: {recall}"
        print(doc_string)
        with open(f'{testing_result_dir}/result.txt', 'a') as out_txt_f:
            out_txt_f.write(doc_string)
        


'''
NO MORE THRESHOLDING :(
'''    
# def test_thresholding(test_loader, threshold_value = 0.4, dir_name_to_save = "thresholding"):
#     testing_result_dir = f'/root/autoencoder_denoiser/testing_real_imgs_results/{test_loader.dataset.data_folder_name}_{dir_name_to_save}/'
#     os.makedirs(testing_result_dir, exist_ok= True)    
#     displayed=0
#     loss = 0
#     orig_SNR, denoised_SNR, SNR_incr = 0,0,0
#     with torch.no_grad():
#         for iter, data in enumerate(tqdm(test_loader)):
#             # if iter==1: break
#             noise, raw,name = data
#             name = "".join([chr(i) for i in name[0]])
#             if len(raw.shape)==3:   
#                 raw, noise = raw.unsqueeze(1), noise.unsqueeze(1)
#             prediction = torch.clone(noise)
#             prediction[abs(prediction)<threshold_value]=0        
#             ground_truth = raw
#             prediction = torch.clip(prediction,-1,1)

#             loss += criterion(prediction,ground_truth )
#             # updating orig_SNR, denoised_SNR, SNR_incr
#             orig_SNR, denoised_SNR, SNR_incr = [sum(x) for x in zip( (orig_SNR, denoised_SNR, SNR_incr ), \
#                                                                      compute_metrics(raw, noise, prediction))] 
    
#         # if need_display and displayed<2:
#             if True:
#                 noise_pic , prediction_pic, raw_pic = noise[0],prediction[0], raw[0]
            
#             # print("?")
#             # plt.clf()

#                 save_path = f'{testing_result_dir}/result_{name}.png'
#                 display_pics(noise_pic[0].cpu(), prediction_pic[0].cpu(), raw_pic[0].cpu(), save_path=save_path)
                
#                 displayed = displayed+1
            
#         loss /= len(test_loader.dataset)  
#         orig_SNR /=   len(test_loader.dataset)
#         denoised_SNR /=   len(test_loader.dataset)
#         SNR_incr /=   len(test_loader.dataset)

#         print("test loader size:" , len(test_loader.dataset))
#         print(f"loss of thresholding is {loss}")
#         print(f"orig_SNR, denoised_SNR, SNR_incr of thresholding is {orig_SNR, denoised_SNR, SNR_incr}")




if __name__ == "__main__":
    # my_model_test = Test("t1_03")
    # paper_1d_test = Test("paper_1d")
    # dann_test = Test('adv', threshold=True)
    
    # dann_test.config['dataset']['real_img_keep_size'] = True 

    # test_thresholding(test_loader, threshold_value = 0.9 , dir_name_to_save="theshold_9_all")


    my_model_test = Test()
    
    test(my_model_test)