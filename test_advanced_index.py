import numpy as np

a = np.zeros((5,5), np.uint8)

b = np.array ([[0,0],
               [0,1],
               [1,0],
               [2,0]])

a[(b.T)] =1
print(a)