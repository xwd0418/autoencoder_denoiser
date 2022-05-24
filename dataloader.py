import os
from sklearn.utils import shuffle
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt

imgs = []
for i in range(1,20):
    data = np.load("dataset/Jeol_info{}000.npy".format(str(i)),allow_pickle=True)
    img_data = data[:,3]
    imgs.append(img_data)
    # the second index has to be 3 to show be some image
all_data  = np.concatenate(imgs)

class HSQC_Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, all_data, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.all_data = all_data
        self.transform = transform

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        sample = self.all_data[idx]

        if self.transform!=None:
            sample = self.transform(sample)

        return sample


def get_datasets(config):
    np.random.shuffle(all_data)
    first = int((all_data)*0.8)
    second = int((all_data)*0.9)

    train = all_data[:first]
    val = all_data[first:second]
    test = all[second:]

    shuffle=config["dataset"]['shuffle']
    batch = config["dataset"]['batch_size']
    train_loader = DataLoader(train, batch_size=batch, shuffle=shuffle, num_workers=4*os.cpu_count)
    val_loader = DataLoader(val, batch_size=batch, shuffle=shuffle, num_workers=4*os.cpu_count)
    test_loader = DataLoader(test, batch_size=batch, shuffle=shuffle, num_workers=4*os.cpu_count)

    return train_loader, val_loader , test_loader


dataset = HSQC_Dataset(all_data)

dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

for i_batch, sample_batched in enumerate(dataloader):
    if i_batch == 3:
        break
    print("batch is",i_batch)
    print(sample_batched.shape)


#### used to show some sample image
# fig = plt.figure()
# for i in range(len(dataset)):
#     sample = dataset[i]

#     print(i, sample.shape)

#     ax = plt.subplot(1, 4, i + 1)
#     plt.tight_layout()
#     ax.set_title('Sample #{}'.format(i))
#     ax.axis('off')
#     plt.imshow(sample)
#     plt.savefig("useless/{}.png".format(str(i)))

#     if i == 3:
#         plt.show()
#         break