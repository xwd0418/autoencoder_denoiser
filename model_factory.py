from logging import raiseExceptions
from matplotlib.pyplot import axis
import numpy as np
import sys
# import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torchvision

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
    elif model_type == "JNet":
        print ("model: JNet (for tessellation)")
        return JNet(1,1,config['model']['bilinear'])
    elif model_type == "UNet_2":
        print ("model: Unet with low-resolution tessellation)")
        return UNet(2,1,config['model']['bilinear'])
    elif model_type == "Adv_UNet":
        print ("model: Adv_Unet")
        return Adv_Unet(1,1,config['model']['bilinear'], config['model']['features'])
    elif model_type == "UNet_Single":
        print ("model: Unet config as the paper indicated)")
        return UNet_Single(1,1,config['model']['bilinear'],config['model']['dim'] )
    else : raise Exception("what is the model to use???")


class JNet(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, bilinear) :
        super(JNet, self).__init__()
        
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.bilinear = bilinear
        
        # the upper part of JNet
        self.inc = DoubleConv(n_channels_in, 64)
        self.down1 = Down(64, 64)
        self.down2 = Down(64, 64)
        
        # the Unet part of Jnet
        # 2 channels: 1 for noise, 1 for pre_filtered tessellation
        self.unet = UNet(1,1,bilinear,tessllation=True) 
        
    def forward(self, x):
        noisy_sample, tessellated_noise = x
        tessellation1 = self.inc(tessellated_noise)
        tessellation2 = self.down1(tessellation1)
        tessellation3 = self.down2(tessellation2)
        
        return self.unet.forward(noisy_sample, tessellate_info=tessellation3)

class UNet(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, bilinear, tessllation = False):
        super(UNet, self).__init__()
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels_in, 64)
        if tessllation:
            self.down1 = Down(128,128)
        else:
            self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_channels_out)

    def forward(self, x, tessellate_info=None, feature=False):
        x1 = self.inc(x)
        if tessellate_info!=None:
            concatenated = torch.concat((x1, tessellate_info), 1)
            x2 = self.down1(concatenated)
        else:
            x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # print('x5 shape is ', x5.shape)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        if feature:
            return logits, x5
        return logits
    
class UNet_Single(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, bilinear, dim):
        super(UNet_Single, self).__init__()
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.bilinear = bilinear
        self.dim = dim
        assert(dim == 1 or dim == 2)
        if dim == 1:
            down_channel = [64, 128, 256, 512,
                            512, 512, 512, 512,
                            512, 512, 512, 512,
                            ]
            up_channel = [512, 512, 512, 512,
                          512, 1024, 1024, 1024,
                          256, 128, 64, 64]
            
        else:
            down_channel = [64, 128, 256, 512,
                            512, 512, 512, 512,
                            512
                            ]
            up_channel = [512, 512, 1024, 1024,
                           1024, 256, 128, 64,
                          64]
        encode_layers = []
        encode_layers.append (SingleConv(n_channels_in, 64, dim=dim))
        for i in range(len(down_channel)-1):
            encode_layers.append(SingleDown(down_channel[i], down_channel[i+1], dim))

        decode_layers = []
        # factor = 2 if bilinear else 1
        for i in range(len(down_channel)-1):
            decode_layers.append(SingleUp(up_channel[i], up_channel[i+1] ,bilinear, dim))

        self.outc = SingleOutConv(64, n_channels_out,dim)
        
        self.encode = nn.Sequential(*encode_layers)
        self.decode = nn.Sequential(*decode_layers)

    def forward(self, x, feature=False):
        if self.dim ==1:
            shape = x.shape
            x = x.view(-1, 1, shape[3])
        
        encode_results=[x]
        for layer in self.encode:
            result = layer(encode_results[-1])
            encode_results.append(result)
        
        # x1 = self.inc(x)
        # if tessellate_info!=None:
        #     concatenated = torch.concat((x1, tessellate_info), 1)
        #     x2 = self.down1(concatenated)
        # else:
        #     x2 = self.down1(x1)
        # x3 = self.down2(x2)
        # x4 = self.down3(x3)
        # x5 = self.down4(x4)
        # # print('x5 shape is ', x5.shape)
        x = encode_results[-1]
        encode_results.pop()
        for layer in self.decode:
            x = layer(x, encode_results[-1])
            encode_results.pop()
            
        logits = self.outc(x)
        if self.dim ==1:
            logits = logits.view(shape)
        if feature:
            raise Exception("shouldn't ask for features here")
        return logits


class Discriminator(nn.Module):
    def __init__(self, feature_nums):
        super(Discriminator, self).__init__()
        self.MLP = MLP(512,feature_nums) # 512 is the num_channel of the encoded infomation
    
    def forward(self, x):
        x = nn.AdaptiveAvgPool2d(1)(x)
        print("after avg pool", x.shape)
        x = self.MLP(x)

class AdversarialNetwork(nn.Module):
  def __init__(self, in_feature, hidden_size):
    super(AdversarialNetwork, self).__init__()
    self.ad_layer1 = nn.Linear(in_feature, hidden_size)
    self.ad_layer1.weight.data.normal_(0, 0.01)
    self.ad_layer1.bias.data.fill_(0.0)

    self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
    self.ad_layer2.weight.data.normal_(0, 0.01)
    self.ad_layer2.bias.data.fill_(0.0)

    self.ad_layer3 = nn.Linear(hidden_size, 1)
    self.ad_layer3.weight.data.normal_(0, 0.3)
    self.ad_layer3.bias.data.fill_(0.0)

    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()
    self.dropout1 = nn.Dropout(0.5)
    self.dropout2 = nn.Dropout(0.5)
    # self.sigmoid = nn.Sigmoid()

  def forward(self, x, coeff):
    x = x * 1.0
    x.register_hook(grl_hook(coeff))
    x = self.ad_layer1(x)
    x = self.relu1(x)
    x = self.dropout1(x)
    x = self.ad_layer2(x)
    x = self.relu2(x)
    x = self.dropout2(x)
    y = self.ad_layer3(x)
    # y = self.sigmoid(y)
    return y


class Adv_Unet(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, bilinear, feature_nums=None):
        super(Adv_Unet,self).__init__()
        self.Unet = UNet(n_channels_in,n_channels_out , bilinear)
        self.discriminator = AdversarialNetwork(512, feature_nums)
    
    def forward(self, x, y=None, coeff=None, plain=True):
        if plain:
            return self.Unet(x)
        combined = torch.cat((x,y))
        combined_results, features = self.Unet(combined, feature=True)
        # features.register_hook(grl_hook(coeff))
        features = nn.AdaptiveAvgPool2d(1)(features).squeeze(2).squeeze(2)
        # print("feature shape",features.shape)

        y = self.discriminator(features, coeff)
        y = nn.Sigmoid()(y)
        return combined_results[:x.shape[0]], y  # only source set denoised_img and all domain predictions

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1





class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
# m = CNN_encoding_model()   



"""deprecated models"""

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



    
 