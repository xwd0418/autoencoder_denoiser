from scipy.spatial import Delaunay
import cv2
import numpy as np


def triangle_tessellate(img1):
    scale_factor = 4 # percent of original size
    width = int(img1.shape[1] * scale_factor )
    height = int(img1.shape[0] * scale_factor )
    dim = (width, height)
    # img1 = cv2.resize(np.array(img1,dtype='uint8'), dim, interpolation = cv2.INTER_AREA)

    # find points
    points = np.array(np.where(img1==1)).T 
    
    #find triangles to draw
    tri = Delaunay(points)                  
    triangles = points[tri.simplices[:]]
    
    triangles = triangles[:,:,[1,0]]     # swap column to match original image
    triangles = scale_factor*triangles   # for better resolution 


    import matplotlib.pyplot as plt

    # Blue color in BGR
    color = (1)
    
    # Line thickness of 2 px
    thickness = 1
    
    # Using cv2.polylines() method
    # Draw a Blue polygon with
    # thickness of 1 px

    output_img = np.zeros(dim, np.uint8)

    for triangle in triangles:
        output_img = cv2.polylines(np.array(output_img), [triangle],
                        True, color, thickness)
        
    return output_img
        