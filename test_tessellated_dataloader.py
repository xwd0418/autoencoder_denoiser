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
    raw_sample, (noisy_sample, tessellated_noise )= data
    break

# print (raw_sample.shape, tessellated_raw.shape, noisy_sample.shape, tessellated_noise.shape)

plt.clf()

ax = plt.subplot(2, 2, 1)
plt.tight_layout()
ax.set_title('orig,{}'.format(raw_sample[0].shape))
ax.axis('off')
plt.imshow(raw_sample[0].cpu())

# ax = plt.subplot(2, 2, 2)
# plt.tight_layout()
# ax.set_title('tessellate,{}'.format(tessellated_raw[0].shape))
# ax.axis('off')
# plt.imshow(tessellated_raw[0].cpu())

ax = plt.subplot(2, 2, 3)
plt.tight_layout()
ax.set_title('noise,{}'.format(noisy_sample[0].shape))
ax.axis('off')
plt.imshow(noisy_sample[0].cpu())

ax = plt.subplot(2, 2, 4)
plt.tight_layout()
ax.set_title('tessellated_noise,{}'.format(tessellated_noise[0].shape))
ax.axis('off')
plt.imshow(tessellated_noise[0].cpu())

plt.savefig("tessellated_dataloader.png")
displayed = True
plt.clf()


