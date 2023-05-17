""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, List, Optional, Sequence, Tuple, Union



class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, oneD=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        conv1  = nn.Conv1d if oneD else nn.Conv2d
        bn1 = nn.BatchNorm1d if oneD else nn.BatchNorm2d
        conv2 = nn.Conv1d if oneD else nn.Conv2d
        bn2 = nn.BatchNorm1d if oneD else nn.BatchNorm2d
        self.double_conv = nn.Sequential(
            conv1(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            bn1(mid_channels),
            nn.ReLU(inplace=True),
            conv2(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            bn2(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class SingleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, dim, stride = 2):
        super().__init__()
        assert (dim==1 or dim==2)
        conv_operation = nn.Conv2d if dim==2 else nn.Conv1d
        batchNorm = nn.BatchNorm2d if dim==2 else nn.BatchNorm1d
        self.conv = nn.Sequential(
            conv_operation(in_channels, out_channels, kernel_size=4, stride=stride, padding=1),
            batchNorm(out_channels),
            nn.LeakyReLU(0.2,inplace=True),
        )

    def forward(self, x):
        # print('going down ',x.shape)
        return self.conv(x)
    
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, oneD=False):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, oneD=oneD)
        ) if not oneD else nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_channels, out_channels, oneD=oneD)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class SingleDown(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, dim):
        super().__init__()
        assert (dim==1 or dim==2)
        # maxpool = nn.MaxPool2d if  dim==2 else nn.MaxPool1d
        self.maxpool_conv = nn.Sequential(
            # maxpool(2),
            SingleConv(in_channels, out_channels, dim)
        )
        # print(f" going down from {in_channels} to {out_channels}")

    def forward(self, x):
        out = self.maxpool_conv(x)
        # print("go down ", out.shape)
        return out

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, oneD=False):
        super().__init__()
        self.oneD =oneD

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, oneD=oneD) 
                
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2) if not oneD else \
                 nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, oneD=oneD)

    def forward(self, x1, x2, skip_top_connection = False):
        x1 = self.up(x1)
        if skip_top_connection:
            return x1
        # input is CHW
        if not self.oneD:
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        else:
            diff = x2.size()[2]-x1.size()[2]
            x1 = F.pad(x1, [diff // 2, diff - diff // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class SingleUp(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, dim=None):
        super().__init__()
        assert (dim==1 or dim==2)
        self.dim = dim
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = SingleConv(in_channels, out_channels, dim, stride=1 )
        else:
            transposeConv =  nn.ConvTranspose2d if  dim==2 else nn.ConvTranspose1d 
            # print(f"go up from channel num {in_channels} to {out_channels}")
            self.up = transposeConv(in_channels, out_channels,padding=1 ,kernel_size=4, stride=2)
            # self.conv = SingleConv(in_channels, out_channels, dim, stride=1)
        self.batchNorm = nn.BatchNorm2d if dim==2 else nn.BatchNorm1d
        self.batchNorm = self.batchNorm(out_channels)
        self.act = nn.LeakyReLU(0.2,inplace=True)


    def forward(self, x1, x2):
        if self.dim ==2:
            x1 = self.up(x1)
            # input is CHW
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            # if you have padding issues, see
            # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
            # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
            x = torch.cat([x2, x1], dim=1)
        else:
            # print('passed shape ', x1.shape)
            x1 = self.up(x1)
            x1 = self.batchNorm(x1)
            x1 = self.act(x1) 
            # print('after up ', x1.shape)
            diff = x2.size()[2] - x1.size()[2]
            x1 = F.pad(x1, [diff // 2, diff - diff // 2])
            x = torch.cat([x2, x1], dim=1)
            # print('padded shape ', x1.shape)
            # print('x2 shape ', x2.shape)
        # print('going up ',x.shape)
        return x
            


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, oneD=False):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if not oneD else \
            nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
class SingleOutConv(nn.Module):
    def __init__(self, in_channels, out_channels, dim):
        super(SingleOutConv, self).__init__()
        assert (dim==1 or dim==2)
        conv_operation = nn.Conv2d if dim==2 else nn.Conv1d

        self.conv = conv_operation(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
    
    
    
    
class MLP(torch.nn.Sequential):
    """This block implements the multi-layer perceptron (MLP) module.

    Args:
        in_channels (int): Number of channels of the input
        hidden_channels (List[int]): List of the hidden channel dimensions
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the convolution layer. If ``None`` this layer wont be used. Default: ``None``
        activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the conv layer. If ``None`` this layer wont be used. Default: ``torch.nn.ReLU``
        inplace (bool): Parameter for the activation layer, which can optionally do the operation in-place. Default ``True``
        bias (bool): Whether to use bias in the linear layer. Default ``True``
        dropout (float): The probability for the dropout layer. Default: 0.0
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        inplace: Optional[bool] = True,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        # The addition of `norm_layer` is inspired from the implementation of TorchMultimodal:
        # https://github.com/facebookresearch/multimodal/blob/5dec8a/torchmultimodal/modules/layers/mlp.py
        params = {} if inplace is None else {"inplace": inplace}

        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(activation_layer(**params))
            layers.append(torch.nn.Dropout(dropout, **params))
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias))
        layers.append(torch.nn.Dropout(dropout, **params))

        super().__init__(*layers)
        # _log_api_usage_once(self)