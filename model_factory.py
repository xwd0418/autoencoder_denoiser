import numpy as np
import sys
import matplotlib.pyplot as plt
import torch.nn as nn
import torch



def get_model(config):
    model_type = config['model']['model_type']
    if model_type == "filter":
        print( "model : filter")
        return Filter()
    else :
        print( "model : auto-encoder")
        return denoising_model()

class denoising_model(nn.Module):
  def __init__(self):
    super(denoising_model,self).__init__()
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


class Filter():
    def __init__(self) -> None:
        self.displayed = False

    def filtering(self,x):
        if x>=0.9: return 1
        return 0
        
    def forward(self, x):
        filtered = np.array([[[self.filtering(float(k)) for k in j] for j in i] for i in x])

        if not self.displayed:
            ax = plt.subplot(1, 4, 2)
            plt.tight_layout()
            ax.set_title('noise')
            ax.axis('off')
            plt.imshow(x[0])

            ax = plt.subplot(1, 4, 3)
            plt.tight_layout()
            ax.set_title('filtered')
            ax.axis('off')
            plt.imshow(filtered[0])
            plt.savefig("useless/compare.png")
            self.displayed = True
        return filtered
        # return np.array ([self.filtering(xi) for xi in x])

    def __call__(self, x):
        return self.forward(x)
