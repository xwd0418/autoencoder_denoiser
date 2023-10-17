import matplotlib.pyplot as plt
import matplotlib.image
import cv2, numpy as np

clist = [(0,"darkblue"), (0.5,"white"), (1, "darkred")]
custom_HSQC_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("_",clist)
diff_clist = [(0,"green"), (0.5,"white"), (1, "red")]
custom_diff_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("_",diff_clist)
plt.rcParams["figure.figsize"] = (16,20)

def display_pics(noise_pic, prediction_pic, raw_pic, title_name=None, save_path=None):
    '''display noisey pic, prediction, clean pic, and difference repestively'''
   
    
    if title_name:
        plt.title(title_name)
   
    ax = plt.subplot(2, 2, 1)
    plt.tight_layout()
    ax.set_title('original',fontsize=18)
                    # ax.axis('off')
    plt.imshow(raw_pic,cmap=custom_HSQC_cmap, vmax=1, vmin=-1)
    plt.colorbar()
                    
    ax = plt.subplot(2, 2, 2)
    plt.tight_layout()
    ax.set_title('noise',fontsize=18)
                    # ax.axis('off')
    plt.imshow(noise_pic,cmap=custom_HSQC_cmap, vmax=1, vmin=-1)
    plt.colorbar()
                    
    ax = plt.subplot(2, 2, 3)
    plt.tight_layout()
    ax.set_title('predicted',fontsize=18)
                    # ax.axis('off')
    plt.imshow(prediction_pic,cmap=custom_HSQC_cmap, vmax=1, vmin=-1)
    plt.colorbar()
                    
    ax = plt.subplot(2, 2, 4)
    plt.tight_layout()
    ax.set_title('difference', fontsize=18)
                    # ax.axis('off')
                    
                    # difference = prediction_pic-raw_pic
                    # difference = difference.float()/2 + 0.5
                    # print(difference)
    difference = cv2.subtract(np.array(prediction_pic), np.array(raw_pic))
    plt.imshow(difference, cmap = custom_diff_cmap, vmax=1, vmin=-1)

    plt.colorbar()
    


                    
    if save_path:
        plt.savefig(save_path)
        plt.clf()
        plt.figure()
        plt.close()
                
    else:
        plt.figure()