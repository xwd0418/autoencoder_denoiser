import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from model1D import *
from utils import *
import scipy.io as io
import numpy as np
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"]="4"

def main_test():
    model_name = tl.global_flag['model']

    # =================================== BASIC CONFIGS =================================== #

    print('[*] run basic configs ... ')

    log_dir = "log_inference_{}".format(model_name)
    tl.files.exists_or_mkdir(log_dir)
    _, _, log_inference, _, _, log_inference_filename = logging_setup(log_dir)

    checkpoint_dir = "checkpoint_inference_{}".format(model_name)
    tl.files.exists_or_mkdir(checkpoint_dir)

    #save_dir = "samples_inference_{}_{}_{}".format(model_name, mask_name, mask_perc)
    #tl.files.exists_or_mkdir(save_dir)

 
    # ==================================== PREPARE DATA ==================================== #



    # ==================================== DEFINE MODEL ==================================== #

    print('[*] define model ... ')

    nw = 1
    nh = 1024*8
    nz  = 1

    print("nw,nh,nz",nw,nh,nz)
    # define placeholders
    t_image_good = tf.placeholder('float32', [None, 1, 8192, 1], name='good_image')
    t_image_bad = tf.placeholder('float32', [None, 1, 8192, 1], name='bad_image')
    t_gen = tf.placeholder('float32', [None, 1, 8192, 1], name='generated_image')

    # define generator network
    if tl.global_flag['model'] == 'unet':
        net_test = u_net_bn(t_image_bad, is_train=False, reuse=False, is_refine=False)
    elif tl.global_flag['model'] == 'unet_refine':
        net_test = u_net_bn(t_image_bad, is_train=False, reuse=False, is_refine=True)
    else:
        raise Exception("unknown model")

    # nmse metric for testing purpose
    nmse_a_0_1 = tf.sqrt(tf.reduce_sum(tf.squared_difference(t_gen, t_image_good), axis=[1, 2, 3]))
    nmse_b_0_1 = tf.sqrt(tf.reduce_sum(tf.square(t_image_good), axis=[1, 2, 3]))
    nmse_0_1 = nmse_a_0_1 / nmse_b_0_1


    
    # ==================================== INFERENCE ==================================== #

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run(tf.global_variables_initializer())
    
    tl.files.load_and_assign_npz(sess=sess,
                                 name=os.path.join(checkpoint_dir, tl.global_flag['model']) + '.npz',
                                 network=net_test)

    X_samples_bad = np.empty((1, 1, 1024*8, 1), dtype=float)
    X_samples_good = np.empty((1, 1, 1024*8, 1), dtype=float)
    
    path0=os.path.abspath('.')
    
    path1=os.path.join(path0, 'NoisySpectrum.mat')

    a = io.loadmat(path1)
    a = a['impure']
    #a = np.transpose(data1)
    a = np.expand_dims(a, axis=2)
    X_samples_bad[0, :, :, :] = a

    path2=os.path.join(path0, 'ReferenceSpectrum.mat')

    b = io.loadmat(path2)
    b = b['pure']
   # b = np.transpose(data2)
    b = np.expand_dims(b, axis=2)
    X_samples_good[0, :, :, :] = b

    x_good_sample_rescaled = X_samples_good 
    x_bad_sample_rescaled = X_samples_bad 



    print('[*] start testing ... ')
    time_start=time.time()

    x_gen = sess.run(net_test.outputs, {t_image_bad: X_samples_bad})
    time_end=time.time()
    print('time_cost',time_end-time_start,'s')

                                                                

    
    Z2=X_samples_bad
    Z2=np.array(Z2,dtype='float64')
    Z2=np.reshape(Z2,(1, 1024*8))
    np.savetxt('BadSampleex.txt', Z2)
    

    Z=X_samples_good
    Z=np.array(Z,dtype='float64')
    Z=np.reshape(Z,(1, 1024*8))
    np.savetxt('GoodSampleex.txt', Z)

    Z1=x_gen
    Z1=np.array(Z1,dtype='float64')
    Z1=np.reshape(Z1,(1, 1024*8))
    np.savetxt('GenerateSampleex.txt', Z1)

    #  plt.show()

   

    print("[*] Job finished!")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='unet', help='unet, unet_refine')

    args = parser.parse_args()

    tl.global_flag['model'] = args.model


    main_test()
