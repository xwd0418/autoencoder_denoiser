import json
import os
from dataloader import get_datasets
from model_factory import get_model
from weakref import ref
import matplotlib.pyplot as plt
import numpy as np
from tessellate import triangle_tessellate
from tqdm import tqdm
from scipy.spatial import Delaunay
import cv2

name = "trivia"
f = open('./hyperparameters/'+ name + '.json')
config = json.load(f)
train_loader, val_loader, test_loader = get_datasets(config)

for iter, data in enumerate(tqdm(train_loader)):
    raw, noise = data
    break

img1 = raw[0]

output_img = triangle_tessellate(img1)

# print (output_img[100])


plt.clf()
plt.imshow(output_img)
plt.savefig("tessellated.png")

plt.clf()
plt.imshow(img1)
plt.savefig("raw0.png")

# print (tri.simplices[:])
