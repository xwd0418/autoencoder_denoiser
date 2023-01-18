from easydict import EasyDict as edict   #可以使得以属性的方式去访问字典的值
import json
import os

config = edict()
config.TRAIN = edict()

config.TRAIN.batch_size = 4 
config.TRAIN.early_stopping_num = 20
config.TRAIN.lr = 0.001
config.TRAIN.lr_decay = 0.01 #learning_rate
config.TRAIN.decay_every = 5
config.TRAIN.beta1 = 0.7  # beta1 in Adam optimiser
config.TRAIN.n_epoch = 5000
config.TRAIN.sample_size = 100
config.TRAIN.g_alpha = 15  # weight for pixel loss
config.TRAIN.g_gamma = 0.0025  # weight for perceptual loss
config.TRAIN.g_beta = 0.1  # weight for frequency loss
config.TRAIN.g_adv = 1  # weight for frequency loss

config.TRAIN.seed = 100
config.TRAIN.epsilon = 0.000001


config.TRAIN.training_data_path = os.path.join('Data1D_zero_filled', 'training')
config.TRAIN.val_data_path = os.path.join( 'Data1D_zero_filled', 'validation')
config.TRAIN.testing_data_path = os.path.join( 'Data1D_zero_filled','testing')
config.TRAIN.training_gooddata_path = os.path.join('Data1D_zero_filled', 'training_good')
config.TRAIN.val_gooddata_path = os.path.join('Data1D_zero_filled', 'validation_good')
config.TRAIN.testing_gooddata_path = os.path.join( 'Data1D_zero_filled', 'testing_good')


def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
