from tensorlayer.layers import *
from utils import *
import tensorlayer as tl



def u_net_bn(x, is_train=False, reuse=False, is_refine=False):

    w_init = tf.truncated_normal_initializer(stddev=0.01)
    b_init = tf.constant_initializer(value=0.0)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("u_net", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        inputs = tf.keras.layers.InputLayer(x, name='input')

        conv1 = Conv2d(inputs, 64, (4, 4), (2, 2), act=None, padding='SAME',
                       W_init=w_init, b_init=b_init, name='conv1')
        conv2 = Conv2d(conv1, 128, (4, 4), (2, 2), act=None, padding='SAME',
                       W_init=w_init, b_init=b_init, name='conv2')
        conv2 = BatchNormLayer(conv2, act=lambda x: tl.act.lrelu(x, 0.2),
                               is_train=is_train, gamma_init=gamma_init, name='bn2')

        conv3 = Conv2d(conv2, 256, (4, 4), (2, 2), act=None, padding='SAME',
                       W_init=w_init, b_init=b_init, name='conv3')
        conv3 = BatchNormLayer(conv3, act=lambda x: tl.act.lrelu(x, 0.2),
                               is_train=is_train, gamma_init=gamma_init, name='bn3')

        conv4 = Conv2d(conv3, 512, (4, 4), (2, 2), act=None, padding='SAME',
                       W_init=w_init, b_init=b_init, name='conv4')
        conv4 = BatchNormLayer(conv4, act=lambda x: tl.act.lrelu(x, 0.2),
                               is_train=is_train, gamma_init=gamma_init, name='bn4')

        conv5 = Conv2d(conv4, 512, (4, 4), (2, 2), act=None, padding='SAME',
                       W_init=w_init, b_init=b_init, name='conv5')
        conv5 = BatchNormLayer(conv5, act=lambda x: tl.act.lrelu(x, 0.2),
                               is_train=is_train, gamma_init=gamma_init, name='bn5')

        conv6 = Conv2d(conv5, 512, (4, 4), (2, 2), act=None, padding='SAME',
                       W_init=w_init, b_init=b_init, name='conv6')
        conv6 = BatchNormLayer(conv6, act=lambda x: tl.act.lrelu(x, 0.2),
                               is_train=is_train, gamma_init=gamma_init, name='bn6')

        conv7 = Conv2d(conv6, 512, (4, 4), (2, 2), act=None, padding='SAME',
                       W_init=w_init, b_init=b_init, name='conv7')
        conv7 = BatchNormLayer(conv7, act=lambda x: tl.act.lrelu(x, 0.2),
                               is_train=is_train, gamma_init=gamma_init, name='bn7')

        conv8 = Conv2d(conv7, 512, (4, 4), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2),
                       padding='SAME', W_init=w_init, b_init=b_init, name='conv8')
        conv8 = BatchNormLayer(conv8, act=lambda x: tl.act.lrelu(x, 0.2),
                               is_train=is_train, gamma_init=gamma_init, name='bn8')
        conv9 = Conv2d(conv8, 512, (4, 4), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2),
                       padding='SAME', W_init=w_init, b_init=b_init, name='conv9')



        up8 = DeConv2d(conv9, 512, (4, 4), out_size=(1, 4), strides=(2, 2), padding='SAME',
                       act=None, W_init=w_init, b_init=b_init, name='deconv8')
        up8 = BatchNormLayer(up8, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init, name='dbn8')

        up7 = ConcatLayer([up8, conv8], concat_dim=3, name='concat7')

        up7 = DeConv2d(up7, 512, (4, 4), out_size=(1, 8), strides=(2, 2), padding='SAME',
                       act=None, W_init=w_init, b_init=b_init, name='deconv7')
        up7 = BatchNormLayer(up7, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init, name='dbn7')

        up6 = ConcatLayer([up7, conv7], concat_dim=3, name='concat6')
        up6 = DeConv2d(up6, 1024, (4, 4), out_size=(1, 16), strides=(2, 2), padding='SAME',
                       act=None, W_init=w_init, b_init=b_init, name='deconv6')
        up6 = BatchNormLayer(up6, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init, name='dbn6')

        up5 = ConcatLayer([up6, conv6], concat_dim=3, name='concat5')
        up5 = DeConv2d(up5, 1024, (4, 4), out_size=(1, 32), strides=(2, 2), padding='SAME',
                       act=None, W_init=w_init, b_init=b_init, name='deconv5')
        up5 = BatchNormLayer(up5, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init, name='dbn5')

        up4 = ConcatLayer([up5, conv5], concat_dim=3, name='concat4')
        up4 = DeConv2d(up4, 1024, (4, 4), out_size=(1, 64), strides=(2, 2), padding='SAME',
                       act=None, W_init=w_init, b_init=b_init, name='deconv4')
        up4 = BatchNormLayer(up4, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init, name='dbn4')

        up3 = ConcatLayer([up4, conv4], concat_dim=3, name='concat3')
        up3 = DeConv2d(up3, 256, (4, 4), out_size=(1, 128), strides=(2, 2), padding='SAME',
                       act=None, W_init=w_init, b_init=b_init, name='deconv3')
        up3 = BatchNormLayer(up3, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init, name='dbn3')

        up2 = ConcatLayer([up3, conv3], concat_dim=3, name='concat2')
        up2 = DeConv2d(up2, 128, (4, 4), out_size=(1, 256), strides=(2, 2), padding='SAME',
                       act=None, W_init=w_init, b_init=b_init, name='deconv2')
        up2 = BatchNormLayer(up2, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init, name='dbn2')

        up1 = ConcatLayer([up2, conv2], concat_dim=3, name='concat1')
        up1 = DeConv2d(up1, 64, (4, 4), out_size=(1, 512), strides=(2, 2), padding='SAME',
                       act=None, W_init=w_init, b_init=b_init, name='deconv1')
        up1 = BatchNormLayer(up1, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init, name='dbn1')

        up0 = ConcatLayer([up1, conv1], concat_dim=3, name='concat0')
        up0 = DeConv2d(up0, 64, (4, 4), out_size=(1, 1024), strides=(2, 2), padding='SAME',
                       act=None, W_init=w_init, b_init=b_init, name='deconv0')
        up0 = BatchNormLayer(up0, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init, name='dbn0')

        if is_refine:
            out = Conv2d(up0, 1, (1, 1), act=tf.nn.tanh, name='out1')
            out = ElementwiseLayer([out, inputs], tf.add, 'add_for_refine') #out = out + inputs
            #out.outputs = tl.act.ramp(out.outputs, v_min=0, v_max=1) #把数值压缩在v_min到v_max之间
        else:
            out = Conv2d(up0, 1, (1, 1), act=lambda x: tl.act.lrelu(x, 0.2), name='out')

    return out



if __name__ == "__main__":
    pass
