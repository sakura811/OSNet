
import numpy as np
import math
import tensorflow as tf
import tensorflow.contrib.slim as slim


__all__ = [
    'osnet_x1_0',
    'osnet_x0_75',
    'osnet_x0_5',
    'osnet_x0_25',
]


class OSNet():
    def __init__(self, x, training, num_classes, depth_channels, focal_loss_flag, IN=False):
        self.training = training
        self.num_classes = num_classes
        self.depth_channels = depth_channels
        self.feat_dim = 512
        self.focal_loss_flag = focal_loss_flag
        self.IN = IN
        self.logits = self.build_OSNet(x)

    ##########
    # Basic layers
    ##########
    def ConvLayer(self, x, out, k=7, s=2, IN=False, name=None):
        x = tf.layers.conv2d(inputs=x, use_bias=False, filters=out, kernel_initializer=tf.compat.v1.keras.initializers.he_normal(), kernel_size=k, strides=s, padding='SAME')

        if IN == True:
            x = slim.instance_norm(x)
        else:
            x = tf.layers.batch_normalization(inputs=x, training=self.training, fused=True, axis=-1, momentum=0.9, epsilon=1e-5)

        x = tf.nn.relu(x)
        return x

    ##########
    # pointwise convilution
    ##########
    def Conv1x1(self, x, out, k=1, s=1, name=None):
        x = tf.layers.conv2d(inputs=x, use_bias=False, filters=out, kernel_initializer=tf.compat.v1.keras.initializers.he_normal(), kernel_size=k, strides=s, padding='VALID')
        x = tf.layers.batch_normalization(inputs=x, training=self.training, fused=True, axis=-1, momentum=0.9, epsilon=1e-5)
        x = tf.nn.relu(x)
        return x

    def Conv1x1Linear(self, x, out, k=1, s=1, name=None):
        x = tf.layers.conv2d(inputs=x, use_bias=False, filters=out, kernel_initializer=tf.compat.v1.keras.initializers.he_normal(), kernel_size=k, strides=s, padding='VALID')
        x = tf.layers.batch_normalization(inputs=x, training=self.training, fused=True, axis=-1, momentum=0.9, epsilon=1e-5)
        return x

    ##########
    # pointwise convilution
    ##########
    def pointwise_conv(self, x, out_channels, kernel=1, stride=1, padding='SAME', name=None):
        in_channels = x.shape.as_list()[3]

        # kaiming uniform initialization.
        maxval = math.sqrt(6.0 / in_channels)

        w = tf.get_variable(name=name + '_pointwise-filter', shape=[kernel, kernel, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_uniform_initializer(-maxval, maxval))
        x = tf.nn.conv2d(input=x, filter=w, strides=[1, stride, stride, 1], padding=padding)
        return x

    ##########
    # depthwise convilution
    ##########
    def depthwise_conv(self, x, out_channels, kernel=3, stride=1, padding='SAME', name=None):
        in_channels = x.shape.as_list()[3]
        if in_channels != out_channels:
            raise AssertionError('in_channels != out_channels', in_channels, out_channels)

        # kaiming uniform initialization.
        maxval = math.sqrt(6.0/in_channels)

        w = tf.get_variable(name=name+'_depthwise-filter', shape=[kernel, kernel, in_channels, 1], dtype=tf.float32, initializer=tf.random_uniform_initializer(-maxval, maxval))

        # filter = [filter_height, filter_width, in_channels, channel_multiplier]
        out = tf.nn.depthwise_conv2d(input=x, filter=w, strides=[1, stride, stride, 1], padding=padding)
        return out


    def LightConv3x3(self, x, out, name=None):
        x = self.pointwise_conv(x, out_channels=out, kernel=1, stride=1, padding='SAME', name=name+'_liteConv')
        x = self.depthwise_conv(x, out_channels=out, kernel=3, stride=1, padding='SAME', name=name+'_liteConv')

        x = tf.layers.batch_normalization(inputs=x, training=self.training, fused=True, axis=-1, momentum=0.9, epsilon=1e-5)
        x = tf.nn.relu(x)
        return x

    ##########
    # AG
    ##########
    def ChannelGate(self, x, channels, reduction=16, name=None):
        b, h, w, c = x.get_shape().as_list()
        ksize = [1, h, w, 1]
        squeeze = tf.nn.avg_pool(x, ksize=ksize, strides=[1, 1, 1, 1], padding='VALID')
        squeeze = tf.reshape(squeeze, [-1, c])

        excitation = tf.layers.dense(inputs=squeeze, use_bias=use_bias, units=channels // reduction, kernel_initializer=tf.compat.v1.keras.initializers.he_normal(), bias_initializer=bias_initializer)

        excitation = tf.nn.relu(excitation)
        
        # fc
        excitation = tf.layers.dense(inputs=excitation, use_bias=use_bias, units=channels, kernel_initializer=tf.compat.v1.keras.initializers.he_normal(), bias_initializer=bias_initializer)

        excitation = tf.nn.sigmoid(excitation)

        excitation = tf.reshape(excitation, [-1, 1, 1, channels])

        scale = x * excitation
        return scale

    ##########
    # Building blocks for omni-scale feature learning
    ##########
    def OSBlock(self, x, out_filters, bottleneck_reduction=4, IN=False, name=None):
        in_filters = x.shape[-1]
        mid_filters = out_filters // bottleneck_reduction
        identity = x

        # Block-Conv1
        x1 = self.Conv1x1(x, mid_filters, name=name+'_conv1')
        print('OSBlock-Conv1: ', x1.get_shape().as_list())

        # Block-Conv2a
        x2a = self.LightConv3x3(x1, mid_filters, name=name+'_conv2a')
        print('OSBlock-Conv2a: ', x2a.get_shape().as_list())

        # Block-Conv2b
        x2b = self.LightConv3x3(x1, mid_filters, name=name+'_conv2b1')
        print('OSBlock-Conv2b1: ', x2b.get_shape().as_list())
        x2b = self.LightConv3x3(x2b, mid_filters, name=name+'_conv2b2')
        print('OSBlock-Conv2b2: ', x2b.get_shape().as_list())

        # Block-Conv2c
        x2c = self.LightConv3x3(x1, mid_filters, name=name+'_conv2c1')
        print('OSBlock-Conv2c1: ', x2c.get_shape().as_list())
        x2c = self.LightConv3x3(x2c, mid_filters, name=name+'_conv2c2')
        print('OSBlock-Conv2c2: ', x2c.get_shape().as_list())
        x2c = self.LightConv3x3(x2c, mid_filters, name=name+'_conv2c3')
        print('OSBlock-Conv2c3: ', x2c.get_shape().as_list())

        # Block-Conv2d
        x2d = self.LightConv3x3(x1, mid_filters, name=name+'_conv2d1')
        print('OSBlock-Conv2d1: ', x2d.get_shape().as_list())
        x2d = self.LightConv3x3(x2d, mid_filters, name=name+'_conv2d2')
        print('OSBlock-Conv2d2: ', x2d.get_shape().as_list())
        x2d = self.LightConv3x3(x2d, mid_filters, name=name+'_conv2d3')
        print('OSBlock-Conv2d3: ', x2d.get_shape().as_list())
        x2d = self.LightConv3x3(x2d, mid_filters, name=name+'_conv2d4')
        print('OSBlock-Conv2d4: ', x2d.get_shape().as_list())

        # Block-AGage-Conv2
        x2a_gate = self.ChannelGate(x2a, mid_filters, name=name+'_1')
        print('OSBlock-AG1: ', x2a_gate.get_shape().as_list())
        x2b_gate = self.ChannelGate(x2b, mid_filters, name=name+'_2')
        print('OSBlock-AG2: ', x2b_gate.get_shape().as_list())
        x2c_gate = self.ChannelGate(x2c, mid_filters, name=name+'_3')
        print('OSBlock-AG3: ', x2c_gate.get_shape().as_list())
        x2d_gate = self.ChannelGate(x2d, mid_filters, name=name+'_4')
        print('OSBlock-AG4: ', x2d_gate.get_shape().as_list())
        x2 = x2a_gate + x2b_gate + x2c_gate + x2d_gate
        print('OSBlock-Conv2: ', x2.get_shape().as_list())

        # Block-Conv3
        x3 = self.Conv1x1Linear(x2, out_filters, name=name+'_conv3')
        print('OSBlock-Conv3: ', x3.get_shape().as_list())

        # downsample
        if in_filters != out_filters:
            identity = self.Conv1x1Linear(identity, out_filters, name=name+'_downsmp')
            print('OSBlock-downsample: ', identity.get_shape().as_list())

        out = x3 + identity

        if IN == True:
            out = slim.instance_norm(out)

        out = tf.nn.relu(out)
        print('OSBlock-relu: ', out.get_shape().as_list())

        return out

    def _osnet_fc(self, input, output, dropout_rate=None):
        v = tf.layers.dense(inputs=input, use_bias=use_bias, units=output, kernel_initializer=tf.compat.v1.keras.initializers.he_normal(), bias_initializer=bias_initializer)

        v = tf.layers.batch_normalization(inputs=v, training=self.training, fused=True, axis=-1, momentum=0.9, epsilon=1e-5)
        v = tf.nn.relu(v)

        if dropout_rate is not None:
            v = tf.layers.dropout(v, dropout_rate)
        return v

    def _classifier(self, input, num_classes):
        y = tf.layers.dense(inputs=input, use_bias=use_bias, units=num_classes, kernel_initializer=tf.compat.v1.keras.initializers.he_normal(), bias_initializer=bias_initializer)
        return y


    ##########
    # Network architecture
    ##########
    def build_OSNet(self, x):
        # layer1
        x = self.ConvLayer(x, out=self.depth_channels[0], IN=self.IN, name='layer1')
        x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")

        # layer2
        x = self.OSBlock(x, out_filters=self.depth_channels[1], IN=self.IN, name='ly2block1')
        x = self.OSBlock(x, out_filters=self.depth_channels[1], IN=self.IN, name='ly2block2')
        x = self.Conv1x1(x, out=self.depth_channels[1], name='ly2trans')

        b, h, w, c = x.get_shape().as_list()
        ksize = [1, h, w, 1]
        x = tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='ly2AVPool')

        # layer3, IN=False
        x = self.OSBlock(x, out_filters=self.depth_channels[2], IN=False, name='ly3block1')
        x = self.OSBlock(x, out_filters=self.depth_channels[2], IN=False, name='ly3block2')
        x = self.Conv1x1(x, out=self.depth_channels[2], name='ly3trans')

        b, h, w, c = x.get_shape().as_list()
        ksize = [1, h, w, 1]
        x = tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='ly3AVPool')

        # layer4, IN=False
        x = self.OSBlock(x, out_filters=self.depth_channels[3], IN=False, name='ly4block1')
        x = self.OSBlock(x, out_filters=self.depth_channels[3], IN=False, name='ly4block2')

        # layer5
        x = self.Conv1x1(x, out=self.depth_channels[3], name='layer5')

        # OSNet-gap
        b, h, w, c = x.get_shape().as_list()
        ksize = [1, h, w, 1]
        v = tf.nn.avg_pool(x, ksize=ksize, strides=[1, 1, 1, 1], padding="VALID")
        v = tf.reshape(v, [-1, c])

        # for OSNet-embedding
        with tf.compat.v1.variable_scope('pre_logit'):
            v = self._osnet_fc(input=v, output=self.feat_dim)
        
        # classifier
        with tf.compat.v1.variable_scope('logit_fc'):  # for softmax_cross_entropy_with_logits_v2, argmax
            y = self._classifier(v, num_classes=self.num_classes)


##########
# Instantiation, depth_channels.
##########
def osnet_x1_0(x, num_classes, training, pretrained=False, focal_loss_flag=False, IN=False, **kwargs):
    # standard size (width x1.0)
    depth_channels = [64, 256, 384, 512]
    model = OSNet(x, training, num_classes, depth_channels, focal_loss_flag, IN=IN)
    return model


def osnet_x0_75(x, num_classes, training, pretrained=False, focal_loss_flag=False, IN=False, **kwargs):
    # medium size (width x0.75)
    depth_channels = [48, 192, 288, 384]
    model = OSNet(x, training, num_classes, depth_channels, focal_loss_flag, IN=IN)
    return model


def osnet_x0_5(x, num_classes, training, pretrained=False, focal_loss_flag=False, IN=False, **kwargs):
    # tiny size (width x0.5)
    depth_channels = [32, 128, 192, 256]
    model = OSNet(x, training, num_classes, depth_channels, focal_loss_flag, IN=IN)
    return model


def osnet_x0_25(x, num_classes, training, pretrained=False, focal_loss_flag=False, IN=False, **kwargs):
    # very tiny size (width x0.25)
    depth_channels = [16, 64, 96, 128]
    model = OSNet(x, training, num_classes, depth_channels, focal_loss_flag, IN=IN)
    return model


