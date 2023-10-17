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
        output_channel = 3 if config["experiment"]["loss_func"] == "CrossEntropy" else 1
        oneD = True if config['model'].get('dim')==1 else False
        return UNet(1,output_channel,config['model']['bilinear'], oneD=oneD)
    
    # #### deprecated models ###
    # elif model_type == "JNet":
    #     print ("model: JNet (for tessellation)")
    #     return JNet(1,1,config['model']['bilinear'])
    # elif model_type == "UNet_2":
    #     print ("model: Unet with low-resolution tessellation)")
    #     return UNet(2,1,config['model']['bilinear'],skip_top_connection = config['model'].get('skip_top_connection') )
    elif model_type == "Adv_UNet":
        print ("model: Adv_Unet")
        softmaxed_output_size = config['model']['output_img_pooling_size'] if config['model']['CDAN'] else 1 
        return Adv_Unet(1,1,config['model']['bilinear'], softmaxed_output_size, config['model']['adv_features'], CDAN=config['model']['CDAN'])
    elif model_type == "UNet_Single":
        print ("model: Unet config as the paper indicated)")
        return UNet_Single(1,1,config['model']['bilinear'],config['model']['dim'], channel_specs=[[64, 128, 256, 512,],[256, 128, 64, 64]])
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
    def __init__(self, n_channels_in, n_channels_out, bilinear, tessllation = False, oneD=False, skip_top_connection = False):
        super(UNet, self).__init__()
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.bilinear = bilinear
        self.skip_top_connection = skip_top_connection
        
        self.inc = DoubleConv(n_channels_in, 64, oneD=oneD)
        if tessllation:
            self.down1 = Down(128,128, oneD=oneD)
        else:
            self.down1 = Down(64, 128, oneD=oneD)
        self.down2 = Down(128, 256, oneD=oneD)
        self.down3 = Down(256, 512, oneD=oneD)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, oneD=oneD)
        self.up1 = Up(1024, 512 // factor, bilinear, oneD=oneD)
        self.up2 = Up(512, 256 // factor, bilinear, oneD=oneD)
        self.up3 = Up(256, 128 // factor, bilinear, oneD=oneD)
        self.up4 = Up(128, 64, bilinear, oneD=oneD)
        self.outc = OutConv(64, n_channels_out, oneD=oneD)

    def forward(self, x, tessellate_info=None, feature=False):
        # print("why is 2",x.shape)
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
        x = self.up4(x, x1, skip_top_connection = self.skip_top_connection)
        logits = self.outc(x)
        if feature:
            return logits, x5
        # print(logits)
        return logits
    
class UNet_Single(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, bilinear, dim, channel_specs=None):
        super(UNet_Single, self).__init__()
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.bilinear = bilinear
        self.dim = dim
        assert(dim == 1 or dim == 2)
        
        if channel_specs==None:
            if dim == 1:
                down_channel = [64, 128, 256, 512,
                                512, 512, 512, 512,
                                512, 512, 512, 512,
                                ]
                up_channel = [512, 512, 512, 512,
                              512, 1024, 1024, 1024,
                              256, 128, 64, 64]
                # up_channel = [512, 256, 128, 64]
                
            else:
                down_channel = [64, 128, 256, 512,
                                512, 512, 512, 512,
                                512
                                ]
                up_channel = [512, 512, 1024, 1024,
                            1024, 256, 128, 64,
                            64]
        else:
            down_channel = channel_specs[0]
            up_channel = channel_specs[1]
            
        # self.intro_conv = nn.Conv1d(n_channels_in, 64, kernel_size=3, padding=1)
        encode_layers = []
        encode_layers.append (SingleConv(1, 64, dim=dim))
        for i in range(len(down_channel)-1):
            # print("channels", down_channel[i], down_channel[i+1])
            encode_layers.append(SingleDown(down_channel[i], down_channel[i+1], dim))

        decode_layers = []
        # factor = 2 if bilinear else 1
        
        decode_layers.append(SingleUp(down_channel[-1], up_channel[0] ,bilinear, dim))
        for i in range(len(down_channel)-1):
            decode_layers.append(SingleUp(up_channel[i]+down_channel[-(i+2)], up_channel[i+1] ,bilinear, dim))

        self.outc = SingleOutConv(64+1, n_channels_out,dim)
        
        self.encode = nn.Sequential(*encode_layers)
        self.decode = nn.Sequential(*decode_layers)

    def forward(self, x, feature=False):
        if self.dim ==1:
            shape = x.shape
            # print('orig shape', shape)
            x = x.view(-1, 1, shape[-1])
        
        # x=self.intro_conv(x)
        # print("after into ", x.shape)
        encode_results=[x]
        for layer in self.encode:
            result = layer(encode_results[-1])
            encode_results.append(result)
        # print('result shape is ', result.shape)
        
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
            
        # print("x shape", x.shape)
        logits = self.outc(x)
        if self.dim ==1:
            # print("logit shape", logits.shape)
            # print('orig shape', shape)
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
        # print("after avg pool", x.shape)
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
    def __init__(self, n_channels_in, n_channels_out, bilinear, pooling_size, feature_out_size=None, CDAN=False):
        super(Adv_Unet,self).__init__()
        self.Unet = UNet(n_channels_in,n_channels_out , bilinear)
        self.discriminator = AdversarialNetwork(512*(pooling_size**2), feature_out_size) 
        #magic num 512c is because of the Unet architecture
        self.CDAN = CDAN
        self.pooling_size = pooling_size
    
    def forward(self, x, y=None, coeff=None, plain=True):
        if plain:
            return self.Unet(x)
        combined = torch.cat((x,y))
        combined_results, features = self.Unet(combined, feature=True)
        # features.register_hook(grl_hook(coeff))
        features = nn.AdaptiveAvgPool2d(1)(features).squeeze(2).squeeze(2)
        
        if self.CDAN == False: # DANN
            y = self.discriminator(features, coeff)
            return combined_results[:x.shape[0]], y, None  # only source set denoised_img and all domain predictions
        
        else: # CDANN    
            pooled_results = nn.AdaptiveAvgPool2d(self.pooling_size)(combined_results)
            pooled_results = pooled_results.view(pooled_results.shape[0], -1)
            softmax_output = torch.softmax(pooled_results, dim=1) 
            # using avg_pooling to reduce dimension
            # softmax_output = softmax_output.view(softmax_output.shape[0], -1)
            op_out = torch.bmm(softmax_output.detach().unsqueeze(2), features.unsqueeze(1))
            y = self.discriminator(op_out.view(-1, softmax_output.size(1) * features.size(1)), coeff)
            return combined_results[:x.shape[0]], y, softmax_output
        
        # print("feature shape",features.shape)

        # y = nn.Sigmoid()(y)  
        """not using sigmoid as we use BCEwithLogitLoss"""

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



    
class CDANLoss(nn.Module):
    ''' Ref: https://github.com/thuml/CDAN/blob/master/pytorch/loss.py
    '''

    def __init__(self, use_entropy=True):
        super(CDANLoss, self).__init__()
        self.use_entropy = use_entropy
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.entropy_loss = EntropyLoss(coeff=1., reduction='none')

    def forward(self, ad_out, softmax_output=None, coeff=1.0, dc_target=None, training=True):
        batch_size = ad_out.shape[0]//2
        if dc_target == None:
           dc_target = torch.cat((torch.ones(batch_size), torch.zeros(ad_out.size(0)-batch_size)), 0).float().to(ad_out.device)
        loss = self.criterion(ad_out.view(-1), dc_target.view(-1))
        # after_sig = nn.Sigmoid()(ad_out).squeeze()
        # loss = nn.BCELoss(reduction='none')(after_sig, dc_target)
        # print("my computed daloss is ",loss)
        if self.use_entropy:
            entropy = self.entropy_loss(softmax_output)
            if training:
                entropy.register_hook(grl_hook(coeff))  # changed this to only hook sign
            entropy = 1.0 + torch.exp(-entropy)
            source_mask = torch.ones_like(entropy)
            source_mask[batch_size:] = 0
            source_weight = entropy * source_mask
            target_mask = torch.ones_like(entropy)
            target_mask[:batch_size] = 0
            target_weight = entropy * target_mask
            weight = source_weight / torch.sum(source_weight).detach().item() + \
                     target_weight / torch.sum(target_weight).detach().item()
            # if training:
            #     weight.register_hook(grl_hook(1))  # changed this to only hook sign
            return coeff*torch.sum(weight * loss) / torch.sum(weight).detach().item()
        else:
            return coeff*torch.mean(loss.squeeze())


class EntropyLoss(nn.Module):
    ''' Ref: https://github.com/thuml/CDAN/blob/master/pytorch/loss.py
    '''

    def __init__(self, coeff=1., reduction='mean'):
        super().__init__()
        self.coeff = coeff
        self.reduction = reduction

    def forward(self, input):

        epsilon = 1e-5
        entropy = -input * torch.log(input + epsilon)
        entropy = torch.sum(entropy, dim=1)
        if self.reduction == 'none':
            return entropy
        return self.coeff * entropy.mean()
    
    
class CustomMSE(nn.Module):
    def __init__(self, weight_false_negative=1) -> None:
        super().__init__()
        self.weight_false_negative = weight_false_negative
        
    def forward(self, prediction,target):
        self.weight = torch.ones(prediction.shape, requires_grad=False).cuda()
        false_nagative_positions = torch.logical_and (torch.abs(prediction)<0.33 , torch.abs(target)>0.67) 
        self.weight[false_nagative_positions] = self.weight_false_negative
        return torch.sum(self.weight * (prediction - target) ** 2)/len(prediction)