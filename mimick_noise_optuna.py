"""
using optuna to mimick noise
"""

from hsqc_dataset import *
from tqdm import tqdm
import json, torch

# real noise part
f = open('/root/autoencoder_denoiser/configs_bitmap/match_hist.json')
config = json.load(f)

config["dataset"]['batch_size'] = 1
EDGE_SIZE = 4
xedges = list(range(0,180+1,EDGE_SIZE))
yedges = list(range(0,120+1,EDGE_SIZE))

real_img_loader = get_real_img_dataset(config)
all_2d_hists= []
for iter, data in enumerate(tqdm(real_img_loader)):
    noise, raw =  data
    curr_n = noise[0,0]
    coords = np.where(curr_n!=0)
    x,y = coords
    H, _, _ = np.histogram2d(x, y, bins=(xedges, yedges), density=True, weights=np.abs(curr_n[coords]))
    # print(H)
    all_2d_hists.append(H)
avg_2d_hist_real_imgs = np.mean(all_2d_hists, axis=0)
# print(avg_2d_hist_real_imgs_kl)

# multiple with 100 to have it sum to one, so it is a proper probability density function 
# cuz we split the image into 10x10 grid, so there are 100 bins/grids/areas for the histogram 
# 
avg_2d_hist_real_imgs_kl = torch.tensor(avg_2d_hist_real_imgs)*(EDGE_SIZE**2)
# print(torch.sum(avg_2d_hist_real_imgs_kl))


# guessing params for fake noise :
import optuna

def objective(trial):
    f = open('/root/autoencoder_denoiser/configs_bitmap/match_hist.json')
    config_to_generate_noise = json.load(f)
    #suggesting
    noise_factor_lower = trial.suggest_float("noise_factor_lower_bonud", 0.00001, 0.5)
    noise_factor_larger = trial.suggest_float("noise_factor_lower_larger", noise_factor_lower, 1)
    config_to_generate_noise['dataset']['noise_factor'] = [noise_factor_lower, noise_factor_larger ]
    
    # config_to_generate_noise['white_noise_rate'] = trial.suggest_float("white_noise_rate", 0.00001, 0.2)
    
    streak_prob_lower = trial.suggest_float("streak_prob_lower_bonud", 0.00001, 0.7)
    streak_prob_larger = trial.suggest_float("streak_prob_lower_larger", streak_prob_lower, 1)
    config_to_generate_noise['dataset']['streak_prob'] = [streak_prob_lower, streak_prob_larger ]
    
    cross_prob_lower = trial.suggest_float("cross_prob_lower_bonud", 0.00001, 0.5)
    cross_prob_larger = trial.suggest_float("cross_prob_lower_larger", cross_prob_lower, 1)
    config_to_generate_noise['dataset']['cross_prob'] = [cross_prob_lower, cross_prob_larger ]
    
    # find avg histogram of fake noise
    all_2d_hists_fake_noise= []
    _,_,test_loader = get_datasets(config_to_generate_noise)
    for iter, data in enumerate(tqdm(test_loader)):
        raw, noise = data
        curr_n = noise[0,0]
        coords = np.where(curr_n!=0)
        x,y = coords
        H, _, _ = np.histogram2d(x, y, bins=(xedges, yedges), density=True,  weights=np.abs(curr_n[coords]))
        all_2d_hists_fake_noise.append(H)
        
        if iter == 60: # because we only have 60 real hsqc examples 
            break
       
    avg_2d_hist_fake_imgs = np.mean(all_2d_hists_fake_noise, axis=0)

    # compute KL-divergence
    avg_2d_hist_fake_imgs_kl = torch.tensor(avg_2d_hist_fake_imgs)*(EDGE_SIZE**2)
    kl_div_func = torch.nn.KLDivLoss(reduction='sum')
    kl_dis = kl_div_func(avg_2d_hist_fake_imgs_kl.log(), avg_2d_hist_real_imgs_kl)
    return kl_dis

if __name__ == "__main__":
    study = optuna.create_study(
        storage='postgresql+psycopg2://testUser:testPassword@10.244.52.139:5432/testDB',  # Specify the storage URL here.
        study_name="mimicking-noise-weighted-by-peak-heights",
        load_if_exists = True
    )
    study.optimize(objective, n_trials=100)
    print(f"Best value: {study.best_value} (params: {study.best_params})")
