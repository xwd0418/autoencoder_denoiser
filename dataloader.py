import os,sys
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

imgs = []
print("loading data ...")
for i in range(1,20):
    data = np.load("dataset/Jeol_info{}000.npy".format(str(i)),allow_pickle=True)
    img_data = data[:,3]
    imgs.append(img_data)
    # the second index has to be 3 to show be some image
all_data  = np.concatenate(imgs)

print("finish loading")

class HSQC_Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, all_data, config=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.all_data = all_data
        self.config = config

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        raw_sample = self.all_data[idx]
        if self.config:
            noise_factor = self.config["dataset"]["noise_factor"]
        else :
            noise_factor = 0.4
        noisy_sample = raw_sample + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=raw_sample.shape)
        noisy_sample = np.clip(noisy_sample, 0., 1.)

        return raw_sample,noisy_sample


def get_datasets(config):
    np.random.shuffle(all_data)
    first = int(len(all_data)*0.8)
    second = int(len(all_data)*0.9)

    train = all_data[:first]
    val = all_data[first:second]
    test = all_data[second:]

    shuffle=config["dataset"]['shuffle']
    batch = config["dataset"]['batch_size']
    train_loader = DataLoader(HSQC_Dataset(train,config), batch_size=batch, shuffle=shuffle, num_workers=os.cpu_count())
    val_loader = DataLoader(HSQC_Dataset(val,config), batch_size=batch, shuffle=shuffle, num_workers=os.cpu_count())
    test_loader = DataLoader(HSQC_Dataset(test,config), batch_size=batch, shuffle=shuffle, num_workers=os.cpu_count())

    return train_loader, val_loader , test_loader





#### used to show some sample image
# dataset = HSQC_Dataset(all_data,)
# dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
# fig = plt.figure()
# for i in range(len(dataset)):
#     sample = dataset[i][0]
#     noise_sample = dataset[i][1]
#     # print(i, sample.shape)

#     ax = plt.subplot(1, 4, 2*i + 1)
#     plt.tight_layout()
#     ax.set_title('Sample #{}'.format(i))
#     ax.axis('off')
#     plt.imshow(sample)

#     ax = plt.subplot(1, 4, 2*i +2 )
#     plt.tight_layout()
#     ax.set_title('noise Sample #{}'.format(i))
#     ax.axis('off')
#     plt.imshow(noise_sample)
#     plt.savefig("useless/{}.png".format(str(i)))

#     if i == 1:
#         plt.show()
#         break

# np.set_printoptions(threshold=sys.maxsize)
# print(dataset[i][1])