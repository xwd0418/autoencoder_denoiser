# originally used to check the computation of accuracy (IoU)
# now it is used to demo some raw/noised/pre_filtered image
import json
import os,sys
from dataloader import get_datasets
from model_factory import get_model
from weakref import ref
import matplotlib.pyplot as plt
import numpy as np
import torch, copy
from tqdm import tqdm
from datetime import datetime
import shutil



name = "only_t1"
f = open('./hyperparameters/'+ name + '.json')
config = json.load(f)
train_loader, val_loader, test_loader = get_datasets(config)

for iter, data in enumerate(tqdm(train_loader)):
    raw, noise = data
    if iter ==2  :
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
plt.imshow(raw[0,0].cpu(),cmap='gray')

ax = plt.subplot(1, 3, 2)
plt.tight_layout()
ax.set_title('noise')
ax.axis('off')
plt.imshow(noise[0,0].cpu(),cmap='gray')

ax = plt.subplot(1, 3, 3)
plt.tight_layout()
ax.set_title('predicted')
ax.axis('off')
plt.imshow(prediction[0,0].cpu(),cmap='gray')

plt.savefig("accu_test_sample.png")
displayed = True
plt.clf()


# print(noise[0,0,28])

