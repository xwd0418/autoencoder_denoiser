import numpy as np
import sys
# import matplotlib.pyplot as plt
import torch.nn as nn
import torch
# import torchvision

import torchvision.models as models
from unet_parts import *


def get_model(config):
    model_type = config['model']['model_type']
    if model_type == "filter":
        print( "model : filter")
        return Filter()
    elif model_type == "MLP":
        print( "model : MLP-auto-encoder")
        return MLP_model()
    elif model_type == "resNet":
        print( "model :resNet encoder")
        return CNN_encoding_model()
    elif model_type == "UNet":
        print("model :UNet")
        return UNet(1,1,config['model']['bilinear'])
    else : raise Exception("what is the model to use???")


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class MLP_model(nn.Module):
  def __init__(self):
    super(MLP_model,self).__init__()
    self.encoder=nn.Sequential(
                  nn.Linear(100*100,256),
                  nn.ReLU(True),
                  nn.Linear(256,128),
                  nn.ReLU(True),
                  nn.Linear(128,64),
                  nn.ReLU(True)
                  )
    
    self.decoder=nn.Sequential(
                  nn.Linear(64,128),
                  nn.ReLU(True),
                  nn.Linear(128,256),
                  nn.ReLU(True),
                  nn.Linear(256,100*100),
                  nn.Sigmoid(),
                  )
    
 
  def forward(self,x):
    x=  torch.flatten(x, start_dim=1)
    # print ("input shape", x.shape)
    # print ("input type", type(x))
    # print ("input data type", x.dtype)

    x=self.encoder(x)
    x=self.decoder(x)
    # print("before reshape",x.shape)
    x=torch.reshape(x,(-1,100,100))
    return x

class CNN_encoding_model(nn.Module):
    def __init__(self):
        super(CNN_encoding_model,self).__init__()
        self.encoder=models.resnet50()
        self.encoder.conv1=torch.nn.Conv2d(1,64,7,stride =2,padding =3,bias=False)
        self.encoder.fc = Identity()
        # print(self.encoder)
        
        self.decoder=nn.Sequential(
                  nn.Linear(2048,4096),
                  nn.ReLU(True),
                  nn.Linear(4096,4096),
                  nn.ReLU(True),
                  nn.Linear(4096,100*100),
                  nn.Sigmoid(),
                  )
    def forward(self,x):

        x=self.encoder(x)
        x=self.decoder(x)
        # print(x.shape)

        # x=self.decoder(x)
        # print("before reshape",x.shape)
        x=torch.reshape(x,(-1,1,100,100))
        return x  
    





class Filter():
    def __init__(self) -> None:
        self.displayed = False

    def filtering(self,x):
        if x>=0.9: return 1
        return 0
        
    def forward(self, x):
        filtered = np.array([[[self.filtering(float(k)) for k in j] for j in i] for i in x])

        # if not self.displayed:
        #     ax = plt.subplot(1, 4, 2)
        #     plt.tight_layout()
        #     ax.set_title('noise')
        #     ax.axis('off')
        #     plt.imshow(x[0])

        #     ax = plt.subplot(1, 4, 3)
        #     plt.tight_layout()
        #     ax.set_title('filtered')
        #     ax.axis('off')
        #     plt.imshow(filtered[0])
        #     plt.savefig("useless/compare.png")
        #     self.displayed = True
        return filtered
        # return np.array ([self.filtering(xi) for xi in x])

    def __call__(self, x):
        return self.forward(x)




class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
m = CNN_encoding_model()    