from hsqc_dataset import *
from tqdm import tqdm
import json
f = open('/root/autoencoder_denoiser/configs_baseline_selection/dann_adv_loss.json')
config = json.load(f)
# test_loader = DataLoader(RealNoiseDataset_Byeol(config), batch_size=2, shuffle=False, num_workers=1)
_,_,test_loader = get_datasets(config)
import matplotlib.pyplot as plt


import matplotlib.image

clist = [(0,"darkblue"), (0.5,"white"), (1, "darkred")]
custom_HSQC_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("_",clist)
raw_image_collection = []
noise_image_collection = []
plt.rcParams["figure.figsize"] = (20,10)
for iter, data in enumerate(tqdm(test_loader)):
    raw, noise = data
    # plt.imshow(raw[0,0],cmap=custom_HSQC_cmap, vmax=1, vmin=-1)
    # plt.figure()
    # plt.imshow(noise[0,0],cmap=custom_HSQC_cmap, vmax=1, vmin=-1)
    # plt.figure()
    if iter == 3: break
print(noise.shape)