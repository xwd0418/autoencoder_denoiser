import numpy as np
import sys
import matplotlib.pyplot as plt
from sqlalchemy import true



def get_model(config):
    model_type = config['model']['model_type']
    if model_type == "filter":
        return Filter()

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
