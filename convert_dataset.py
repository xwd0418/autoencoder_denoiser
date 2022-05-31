import os,sys
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

imgs = []

for i in range(1,20):
    data = np.load("dataset/Jeol_info{}000.npy".format(str(i)),allow_pickle=True)
    img_data = data[:,3]
    imgs.append(img_data)
    # the second index has to be 3 to show be some image
all_data  = np.concatenate(imgs)

np.save('dataset/single.npy', all_data)