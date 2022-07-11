import numpy as np
import copy

a = np.zeros((5,5), np.uint8)

b = np.array ([[2,0],
               [3,1],
               [4,0],
               [1,2]])

a[tuple(b.T)] =1 
print("a is \n", a)
print("\n\n")
points =  np.array(np.where(a==1))

points_copy = copy.deepcopy(points)
points_copy[1] += 1
points = np.concatenate((points, points_copy),1)
print(points)

noise  = np.zeros((5,5))
noise[tuple(points)] =0.3


print("noise is \n ", noise)
