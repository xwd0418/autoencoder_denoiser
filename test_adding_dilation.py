import json
import os
from weakref import ref
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import shutil
import matplotlib.image
from glob import glob
import torch
from model_factory import UNet

def change_dilation(model):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            change_dilation(module)
        if isinstance(module, torch.nn.Conv2d):
            module.dilation = (2,2)

model = UNet(1,1,True) 
change_dilation(model)
print(model)


            