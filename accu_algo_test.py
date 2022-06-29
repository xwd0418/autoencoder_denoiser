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

name = "trivia"
f = open('./hyperparameters/'+ name + '.json')
config = json.load(f)
train_loader, val_loader, test_loader = get_datasets(config)

for iter, data in enumerate(tqdm(train_loader)):
    raw, noise = data
    break

prediction = noise.round()
intersec = np.sum(np.array(raw.cpu()) * np.array(prediction.cpu()))
union = torch.sum(raw)+torch.sum(prediction)-intersec
accu =intersec / union

print("raw is ", torch.sum(raw))
print("predict is ",torch.sum(prediction) )
print("interesct is ", intersec )
print("union is " , union)

print(accu)

plt.clf()

ax = plt.subplot(1, 3, 1)
plt.tight_layout()
ax.set_title('orig')
ax.axis('off')
plt.imshow(raw[0].cpu())

ax = plt.subplot(1, 3, 2)
plt.tight_layout()
ax.set_title('noise')
ax.axis('off')
plt.imshow(noise[0].cpu())

ax = plt.subplot(1, 3, 3)
plt.tight_layout()
ax.set_title('predicted')
ax.axis('off')
plt.imshow(prediction[0].cpu())

plt.savefig("accu_test_sample.png")
displayed = True
plt.clf()


