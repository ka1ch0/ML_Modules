from typing import Tuple

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, \
    AveragePooling2D, Activation, BatchNormalization, ZeroPadding2D, Reshape
from tensorflow.keras.models import Model


class dw_pw_conv(Model):
    """Mobile Net architectur depthwise + pointwise conv block"""

    def __init__(self, pw_filters: int,
                 activation: str = 'relu',
                 dw_kernel: Tuple[int] = (3, 3),
                 pw_kernel: Tuple[int] = (1, 1),
                 dw_strides: int = 2,
                 pw_strides: int = 1,
                 dw_padding: str = 'same',
                 pw_padding: str = 'same',
                 alpha: float = 1.0):
        super().__init__()
        self.dw_conv = DepthwiseConv2D(kernel_size=dw_kernel,
                                       strides=dw_strides,
                                       padding=dw_padding)
        self.dw_bn = BatchNormalization()
        self.dw_activation = Activation(activation)
        self.pw_conv = Conv2D(filters=pw_filters*alpha, kernel_size=pw_kernel,
                              strides=pw_strides, padding=pw_padding)
        self.pw_bn = BatchNormalization()
        self.pw_activation = Activation(activation)

    def call(self, x):
        h = self.dw_conv(x)
        h = self.dw_bn(h)
        h = self.dw_activation(h)
        h = self.pw_conv(h)
        h = self.pw_bn(h)
        y = self.pw_activation(h)
        return y


class MBNetv1(Model):
    def __init__(self, channels: int,
                 output_shape: int,
                 activation: str = 'relu',
                 alpha: float = 1.0,
                 rho: int = 224):
        super().__init__()
        self.in_shape = [rho, rho, channels]
        self.out_shape = output_shape
        self.activation = activation
        self.alpha = alpha
        self.rho = rho
        self.model_layers = [Input(shape=self.in_shape),
                             Conv2D(filters=32, kernel_size=(3, 3),
                                    strides=2, padding='valid'),
                             BatchNormalization(),
                             Activation(self.activation),
                             dw_pw_conv(64, dw_strides=1),
                             dw_pw_conv(128, dw_strides=2),
                             dw_pw_conv(128, dw_strides=1),
                             dw_pw_conv(256, dw_strides=2),
                             dw_pw_conv(256, dw_strides=1),
                             dw_pw_conv(512, dw_strides=2),
                             dw_pw_conv(512, dw_strides=1),
                             dw_pw_conv(512, dw_strides=1),
                             dw_pw_conv(512, dw_strides=1),
                             dw_pw_conv(512, dw_strides=1),
                             dw_pw_conv(512, dw_strides=1),
                             dw_pw_conv(1024, dw_strides=2),
                             ZeroPadding2D(padding=(3, 3)),
                             dw_pw_conv(1024, dw_strides=2),
                             AveragePooling2D(pool_size=(7, 7)),
                             Conv2D(filters=1000, kernel_size=(1, 1),
                                    strides=1, padding='same')]

    def __call__(self, x):
        for layer in self.model_layers:
            x = layer(x)
        out = Activation('softmax')
        out = Reshape([self.out_shape])
        return out