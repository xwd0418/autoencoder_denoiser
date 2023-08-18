import json
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import json
import cv2
import matplotlib.image
from glob import glob
import torch
from model_factory import get_model
import argparse
from model_factory import UNet
from hsqc_dataset import * 
from tqdm import tqdm

f = open('/root/autoencoder_denoiser/configs_baseline_selection/baseline.json')
config = json.load(f)
train_loader, val_loader, test_loader = get_datasets(config)


from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = (20,10)
clist = [(0,"darkblue"), (0.5,"white"), (1, "darkred")]
custom_HSQC_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("_",clist)

for iter, data in enumerate(tqdm(train_loader)):
    raw, noise = (data)