"""implements many loss functions, refering to this link : 
https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU
    
    
ALPHA = 0.5
BETA = 0.5

class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
       
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        
        return 1 - Tversky
    
    

class DiceLossWithLogits(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLossWithLogits, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice
    
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE
    
class SoftF1ScoreLoss(nn.Module):
    """
    soft version of f1 
    refer to https://stackoverflow.com/questions/65318064/can-i-trainoptimize-on-f1-score-loss-with-pytorch
    """
    def __init__(self):
        super(SoftF1ScoreLoss, self).__init__()
        
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)    
        TP = torch.sum(inputs*targets)
        FP = torch.sum(inputs*(1-targets))
        FN = torch.sum((1-inputs)*targets)
        soft_F1 = TP/(TP+0.5*(FP+FN))
        return 1-soft_F1
    
    
def get_loss_func(config):
    if  config["experiment"]["loss_func"] == "MSE":
        criterion = torch.nn.MSELoss()
    elif config["experiment"]["loss_func"] == "CrossEntropy":
        criterion = torch.nn.CrossEntropyLoss() # edited
    elif config["experiment"]["loss_func"] == "BCE":
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([config['experiment']['BCE_pos_weight']])) # edited
    elif  config["experiment"]["loss_func"] == "Jaccard":
        criterion = IoULoss()
    elif  config["experiment"]["loss_func"] == "SoftF1":
        criterion = SoftF1ScoreLoss()
    elif  config["experiment"]["loss_func"] == "DiceBCE":
        criterion = DiceBCELoss()
    elif  config["experiment"]["loss_func"] == "Dice":
        criterion = DiceLossWithLogits()
    elif  config["experiment"]["loss_func"] == "Tversky":
        criterion = TverskyLoss()
    else:
        raise Exception("what is your loss function??")
    
    return criterion