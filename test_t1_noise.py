from dataloader import *
import json

name = "trivia"
f = open('./hyperparameters/'+ name + '.json')
config = json.load(f)

a = np.zeros((5,5),np.intc)
a[2,3] = 1
a= add_t1_noise(a, config)


print(a)