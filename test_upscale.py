from scipy.spatial import Delaunay
import cv2
import numpy as np


a = np.zeros((3,4))
a[1,2] = 1

print(a)

img1 = cv2.resize(np.array(a,dtype='uint64'), (a.shape[1]*2,a.shape[0]*2), interpolation = cv2.INTER_AREA) 
print(img1)