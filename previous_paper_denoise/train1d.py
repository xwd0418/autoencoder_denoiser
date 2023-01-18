import pickle
from model1D import *
from utils import *
import tensorlayer as tl
from config import config, log_config
from scipy.io import loadmat, savemat
import os
import time
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]="3"


def main_train():
    model_name = tl.global_flag['model']    #模型

    # =================================== BASIC CONFIGS =================================== #

    print('[*] run basic configs ... ')

    log_dir = "log_{}".format(model_name)        #定义指定格式的文件夹
    tl.files.exists_or_mkdir(log_dir)                                        #创建指定文件夹
    log_all, log_eval, log_50, log_all_filename, log_eval_filename, log_50_filename = logging_setup(log_dir)

    checkpoint_dir = "checkpoint_{}".format(model_name)
    tl.files.exists_or_mkdir(checkpoint_dir)

    # configs
    batch_size = config.TRAIN.batch_size
    early_stopping_num = config.TRAIN.early_stopping_num
    g_alpha = config.TRAIN.g_alpha
    g_beta = config.TRAIN.g_beta
    g_adv = config.TRAIN.g_adv
    lr = config.TRAIN.lr
    lr_decay = config.TRAIN.lr_decay   #学习率衰减
    decay_every = config.TRAIN.decay_every
    beta1 = config.TRAIN.beta1
    n_epoch = config.TRAIN.n_epoch
    sample_size = config.TRAIN.sample_size

    log_config(log_all_filename, config)
    log_config(log_eval_filename, config)
    log_config(log_50_filename, config)

    

    # ==================================== PREPARE DATA ==================================== #

    print('[*] load data ... ')
    training_data_path = config.TRAIN.training_data_path
    val_data_path = config.TRAIN.val_data_path
   # testing_data_path = config.TRAIN.testing_data_path

    training_gooddata_path = config.TRAIN.training_gooddata_path
    val_gooddata_path = config.TRAIN.val_gooddata_path
   # testing_gooddata_path = config.TRAIN.testing_gooddata_path


    # ==================================== DEFINE MODEL ==================================== #

    print('[*] define model ... ')
    nh=1024*8


    # define placeholders
    t_FFT_good = tf.placeholder('float32', [None, 1, nh, 1], name='good_image')     #正样本
    t_FFT_bad = tf.placeholder('float32', [None, 1, nh, 1], name='bad_image')       #负样本
    t_gen = tf.placeholder('float32', [None, 1, nh, 1], name='generate_image')  # gen样本


    # define generator network
    if tl.global_flag['model'] == 'unet':
        net = u_net_bn(t_FFT_bad, is_train=True, reuse=False, is_refine=False)
        net_test = u_net_bn(t_FFT_bad, is_train=False, reuse=True, is_refine=False)
        #net_test_sample = u_net_bn(t_FFT_bad_samples, is_train=False, reuse=True, is_refine=False)

    elif tl.global_flag['model'] == 'unet_refine':
        net = u_net_bn(t_FFT_bad, is_train=True, reuse=False, is_refine=True)
        net_test = u_net_bn(t_FFT_bad, is_train=False, reuse=True, is_refine=True)
        #net_test_sample = u_net_bn(t_FFT_bad_samples, is_train=False, reuse=True, is_refine=True)
    else:
        raise Exception("unknown model")

   



    # ==================================== DEFINE LOSS ==================================== #

    print('[*] define loss functions ... ')



    # generator loss (pixel-wise)
    g_nmse_a = tf.sqrt(tf.reduce_sum(tf.squared_difference(net.outputs, t_FFT_good), axis=[1, 2, 3]))
    g_nmse_b = tf.sqrt(tf.reduce_sum(tf.square(t_FFT_good), axis=[1, 2, 3]))
    g_loss = tf.reduce_mean(g_nmse_a / g_nmse_b)

    nmse_a_0_1 = tf.sqrt(tf.reduce_sum(tf.squared_difference(t_gen, t_FFT_good), axis=[1, 2, 3]))
    nmse_b_0_1 = tf.sqrt(tf.reduce_sum(tf.square(t_FFT_good), axis=[1, 2, 3]))
    nmse_0_1 = nmse_a_0_1 / nmse_b_0_1



    # ==================================== DEFINE TRAIN OPTS ==================================== #

    print('[*] define training options ... ')

    g_vars = tl.layers.get_variables_with_name('u_net', True, True)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr, trainable=False)

    g_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss, var_list=g_vars)

    # ==================================== TRAINING ==================================== #

    config1 = tf.ConfigProto(allow_soft_placement=True)
    config1.gpu_options.allow_growth = True
    sess = tf.Session(config=config1)
    tl.layers.initialize_global_variables(sess)

    # load generator and discriminator weights (for continuous training purpose)
    tl.files.load_and_assign_npz(sess=sess,
                                 name=os.path.join(checkpoint_dir, tl.global_flag['model']) + '.npz',
                                 network=net)




    n_training_examples = len([name for name in os.listdir(training_data_path) if os.path.isfile(os.path.join(training_data_path, name))])
    n_step_epoch = round(n_training_examples / batch_size)

    n_validation_examples = len([name for name in os.listdir(val_data_path) if os.path.isfile(os.path.join(val_data_path, name))])
    n_validation_step_epoch = round(n_validation_examples / batch_size)





    print('[*] start training ... ')

    best_nmse = np.inf
    best_epoch = 1
    esn = early_stopping_num
    STEP = 0
    gloss = []
    for epoch in range(0, n_epoch):

        # learning rate decay
        if epoch != 0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay ** (epoch / decay_every)
            sess.run(tf.assign(lr_v, lr * new_lr_decay))
            log = " ** new learning rate: %0.16f" % (lr * new_lr_decay)
            print(log)
            log_all.debug(log)
        elif epoch == 0:
            log = " ** init lr: %0.16f  decay_every_epoch: %d, lr_decay: %0.16f" % (lr, decay_every, lr_decay)
            print(log)
            log_all.debug(log)

        for step in range(n_step_epoch):
            STEP += 1
            step_time = time.time()
            idex = tl.utils.get_random_int(min=1, max=n_training_examples, number=batch_size)
            idrx = tl.utils.get_random_int(min=0, max=3, number=4)

            seq1 = ('impure', str(idex[0]), '.txt')
            seq2 = ('impure', str(idex[1]), '.txt')
            seq3 = ('impure', str(idex[2]), '.txt')
            seq4 = ('impure', str(idex[3]), '.txt')
            A1 = ''.join(seq1)
            A2 = ''.join(seq2)
            A3 = ''.join(seq3)
            A4 = ''.join(seq4)

            X_bad1 = np.loadtxt(os.path.join(training_data_path,A1))
            X_bad1 = X_bad1[idrx[0]]
            X_bad1 = np.expand_dims(X_bad1, axis=2)
            X_bad2 = np.loadtxt(os.path.join(training_data_path,A2))
            X_bad2 = X_bad2[idrx[1]]
            X_bad2 = np.expand_dims(X_bad2, axis=2)
            X_bad3 = np.loadtxt(os.path.join(training_data_path,A3))
            X_bad3 = X_bad3[idrx[2]]
            X_bad3 = np.expand_dims(X_bad3, axis=2)
            X_bad4 = np.loadtxt(os.path.join(training_data_path,A4))
            X_bad4 = X_bad4[idrx[3]]
            X_bad4 = np.expand_dims(X_bad4, axis=2)
            X_bad = np.empty((batch_size,1, nh, 1), dtype=float)
            X_bad[0, :, :, :] = X_bad1
            X_bad[1, :, :, :] = X_bad2
            X_bad[2, :, :, :] = X_bad3
            X_bad[3, :, :, :] = X_bad4

            saq1 = ('pure', str(idex[0]), '.txt')
            saq2 = ('pure', str(idex[1]), '.txt')
            saq3 = ('pure', str(idex[2]), '.txt')
            saq4 = ('pure', str(idex[3]), '.txt')
            B1 = ''.join(saq1)
            B2 = ''.join(saq2)
            B3 = ''.join(saq3)
            B4 = ''.join(saq4)

            X_good1 = np.loadtxt(os.path.join(training_gooddata_path,B1))
            X_good1 = np.expand_dims(X_good1, axis=2)
            X_good2 = np.loadtxt(os.path.join(training_gooddata_path,B2))
            X_good2 = np.expand_dims(X_good2, axis=2)
            X_good3 = np.loadtxt(os.path.join(training_gooddata_path,B3))
            X_good3 = np.expand_dims(X_good3, axis=2)
            X_good4 = np.loadtxt(os.path.join(training_gooddata_path,B4))
            X_good4 = np.expand_dims(X_good4, axis=2)
            X_good = np.empty((batch_size,1, nh, 1), dtype=float)
            X_good[0, :, :, :] = X_good1
            X_good[1, :, :, :] = X_good2
            X_good[2, :, :, :] = X_good3
            X_good[3, :, :, :] = X_good4


            errG, _ = sess.run([g_loss, g_optim], {
                                                   t_FFT_good: X_good,
                                                   t_FFT_bad: X_bad})
            gloss.append(errG)
            if STEP%300==0: 
                np.savetxt('g_loss'+'%d.txt'%STEP, gloss)

            log = "Epoch[{:3}/{:3}] step={:3} g_loss={:5} took {:3}s".format(
                epoch + 1,
                n_epoch,
                step,
                round(float(errG), 3),
                round(time.time() - step_time, 2))
            print(log)
            log_all.debug(log)
            temp=[]
            errG, _ = sess.run([g_loss, g_optim], {
                                                   t_FFT_good: X_good,
                                                   t_FFT_bad: X_bad})

            if STEP%100==0: 
               tl.files.save_npz(net.all_params,name=os.path.join(checkpoint_dir, tl.global_flag['model']) + '.npz',sess=sess)
            
        # evaluation for training data
        total_nmse_training = 0
        num_training_temp = 0
        for step in range(n_validation_step_epoch):
            idex = tl.utils.get_random_int(min=1, max=n_training_examples, number=batch_size)
            idrx = tl.utils.get_random_int(min=0, max=3, number=4)

            seq1 = ('impure', str(idex[0]), '.txt')
            seq2 = ('impure', str(idex[1]), '.txt')
            seq3 = ('impure', str(idex[2]), '.txt')
            seq4 = ('impure', str(idex[3]), '.txt')
            A1 = ''.join(seq1)
            A2 = ''.join(seq2)
            A3 = ''.join(seq3)
            A4 = ''.join(seq4)

            X_bad1 = np.loadtxt(os.path.join(training_data_path,A1))
            X_bad1 = X_bad1[idrx[0]]
            X_bad1 = np.expand_dims(X_bad1, axis=2)
            X_bad2 = np.loadtxt(os.path.join(training_data_path,A2))
            X_bad2 = X_bad2[idrx[1]]
            X_bad2 = np.expand_dims(X_bad2, axis=2)
            X_bad3 = np.loadtxt(os.path.join(training_data_path,A3))
            X_bad3 = X_bad3[idrx[2]]
            X_bad3 = np.expand_dims(X_bad3, axis=2)
            X_bad4 = np.loadtxt(os.path.join(training_data_path,A4))
            X_bad4 = X_bad4[idrx[3]]
            X_bad4 = np.expand_dims(X_bad4, axis=2)
            X_bad = np.empty((batch_size,1, nh, 1), dtype=float)
            X_bad[0, :, :, :] = X_bad1
            X_bad[1, :, :, :] = X_bad2
            X_bad[2, :, :, :] = X_bad3
            X_bad[3, :, :, :] = X_bad4

            saq1 = ('pure', str(idex[0]), '.txt')
            saq2 = ('pure', str(idex[1]), '.txt')
            saq3 = ('pure', str(idex[2]), '.txt')
            saq4 = ('pure', str(idex[3]), '.txt')
            B1 = ''.join(saq1)
            B2 = ''.join(saq2)
            B3 = ''.join(saq3)
            B4 = ''.join(saq4)

            X_good1 = np.loadtxt(os.path.join(training_gooddata_path, B1))
            X_good1 = np.expand_dims(X_good1, axis=2)
            X_good2 = np.loadtxt(os.path.join(training_gooddata_path, B2))
            X_good2 = np.expand_dims(X_good2, axis=2)
            X_good3 = np.loadtxt(os.path.join(training_gooddata_path, B3))
            X_good3 = np.expand_dims(X_good3, axis=2)
            X_good4 = np.loadtxt(os.path.join(training_gooddata_path, B4))
            X_good4 = np.expand_dims(X_good4, axis=2)
            X_good = np.empty((batch_size, 1, nh, 1), dtype=float)
            X_good[0, :, :, :] = X_good1
            X_good[1, :, :, :] = X_good2
            X_good[2, :, :, :] = X_good3
            X_good[3, :, :, :] = X_good4
            x_gen = sess.run(net_test.outputs, {t_FFT_bad: X_bad})

            x_gen_0_1 = x_gen
            x_good_0_1 = X_good

            nmse_res = sess.run(nmse_0_1, {t_gen: x_gen_0_1, t_FFT_good: x_good_0_1})
            total_nmse_training += np.sum(nmse_res)
            num_training_temp += batch_size

        total_nmse_training /= num_training_temp

        log = "Epoch: {}\nNMSE training: {:8}".format(
            epoch + 1,
            total_nmse_training,
             )
        print(log)
        log_all.debug(log)
        log_eval.info(log)

        # evaluation for validation data
        total_nmse_val = 0
        num_val_temp = 0
        for step in range(n_validation_step_epoch):
            idex = tl.utils.get_random_int(min=16001, max=16000+n_validation_examples, number=batch_size)
            idrx = tl.utils.get_random_int(min=0, max=3, number=4)

            seq1 = ('impure', str(idex[0]), '.txt')
            seq2 = ('impure', str(idex[1]), '.txt')
            seq3 = ('impure', str(idex[2]), '.txt')
            seq4 = ('impure', str(idex[3]), '.txt')
            A1 = ''.join(seq1)
            A2 = ''.join(seq2)
            A3 = ''.join(seq3)
            A4 = ''.join(seq4)

            X_bad1 = np.loadtxt(os.path.join(val_data_path,A1))
            X_bad1 = X_bad1[idrx[0]]
            X_bad1 = np.expand_dims(X_bad1, axis=2)
            X_bad2 = np.loadtxt(os.path.join(val_data_path,A2))
            X_bad2 = X_bad2[idrx[1]]
            X_bad2 = np.expand_dims(X_bad2, axis=2)
            X_bad3 = np.loadtxt(os.path.join(val_data_path,A3))
            X_bad3 = X_bad3[idrx[2]]
            X_bad3 = np.expand_dims(X_bad3, axis=2)
            X_bad4 = np.loadtxt(os.path.join(val_data_path,A4))
            X_bad4 = X_bad4[idrx[3]]
            X_bad4 = np.expand_dims(X_bad4, axis=2)
            X_bad = np.empty((batch_size,1, nh, 1), dtype=float)
            X_bad[0, :, :, :] = X_bad1
            X_bad[1, :, :, :] = X_bad2
            X_bad[2, :, :, :] = X_bad3
            X_bad[3, :, :, :] = X_bad4

            saq1 = ('pure', str(idex[0]), '.txt')
            saq2 = ('pure', str(idex[1]), '.txt')
            saq3 = ('pure', str(idex[2]), '.txt')
            saq4 = ('pure', str(idex[3]), '.txt')
            B1 = ''.join(saq1)
            B2 = ''.join(saq2)
            B3 = ''.join(saq3)
            B4 = ''.join(saq4)

            X_good1 = np.loadtxt(os.path.join(val_gooddata_path,B1))
            X_good1 = np.expand_dims(X_good1, axis=2)
            X_good2 = np.loadtxt(os.path.join(val_gooddata_path,B2))
            X_good2 = np.expand_dims(X_good2, axis=2)
            X_good3 = np.loadtxt(os.path.join(val_gooddata_path,B3))
            X_good3 = np.expand_dims(X_good3, axis=2)
            X_good4 = np.loadtxt(os.path.join(val_gooddata_path,B4))
            X_good4 = np.expand_dims(X_good4, axis=2)
            X_good = np.empty((batch_size, 1, nh, 1), dtype=float)
            X_good[0, :, :, :] = X_good1
            X_good[1, :, :, :] = X_good2
            X_good[2, :, :, :] = X_good3
            X_good[3, :, :, :] = X_good4
            x_gen = sess.run(net_test.outputs, {t_FFT_bad: X_bad})
            nmse_res = sess.run(nmse_0_1, {t_gen: x_gen, t_FFT_good: X_good})
            total_nmse_val += np.sum(nmse_res)
            num_val_temp += (batch_size*2)
            
            temp = []
    
        total_nmse_val /= num_val_temp
        log = "Epoch: {}\nNMSE val: {:8}".format(
            epoch + 1,
            total_nmse_val,
            )
        print(log)
        log_all.debug(log)
        log_eval.info(log)


        if total_nmse_val < best_nmse:
            esn = early_stopping_num  # reset early stopping num
            best_nmse = total_nmse_val
            best_epoch = epoch + 1

            # save current best model
            tl.files.save_npz(net.all_params,
                              name=os.path.join(checkpoint_dir, tl.global_flag['model']) + '.npz',
                              sess=sess)

            print("[*] Save checkpoints SUCCESS!")
        else:
            esn -= 1

        log = "Best NMSE result: {} at {} epoch".format(best_nmse, best_epoch)
        log_eval.info(log)
        log_all.debug(log)
        print(log)

        



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='unet', help='unet, unet_refine')

    args = parser.parse_args()

    tl.global_flag['model'] = args.model

    main_train()
