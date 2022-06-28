import json
import os
# from dataloader import get_datasets
from model_factory import get_model
import numpy as np
import torch, copy
import torch.nn as nn
from tqdm import tqdm


a = torch.tensor([
    [0.1, 0.9],
    [0.81 , 0.8],
    [1,   1.5]
    ]).float()
b = torch.tensor([0,1,1]).long()
criterion = torch.nn.CrossEntropyLoss()
print(criterion(a,b))







# batch_size = 3
# nb_classes = 2
# in_features = 10

# model = nn.Linear(in_features, nb_classes)
# criterion = nn.CrossEntropyLoss()

# x = torch.randn(batch_size, in_features)
# target = torch.empty(batch_size, dtype=torch.long).random_(nb_classes)

# output = model(x)
# loss = criterion(output, target)

# print(x)
# print("x type", x.dtype)
# print(target)
# print("target type", target.dtype)

# print(loss)
