"""
@author: Ming Ming Zhang, mmzhangist@gmail.com

ProtoNet Backbone
"""

import tensorflow as tf


def conv_block(inputs, filters, block, train_bn=False):
    """
    Builds a convolution block of ProtoNet.

    Parameters
    ----------
    inputs : tf tensor, [batch_size, height, width, channels]
        An input tensor.
    filters : integer
        The number of filters/units in conv layer.
    block : string
        The name of the block.
    train_bn : boolean, optional
        Whether one should normalize the layer input by the mean and variance 
        over the current batch. The default is False, i.e., use the moving
        average of mean and variance to normalize the layer input.

    Returns
    -------
    x : tf tensor, [batch_size, height//2, width//2, filters] 
        The output tensor.

    """
    conv_name = 'conv_' + block
    bn_name = 'bn_' + block
    act_name = 'relu_' + block
    pool_name = 'maxpool_' + block
    
    x = tf.keras.layers.Conv2D(
        filters, 
        (3, 3), 
        padding='same', 
        name=conv_name)(inputs)
    x = tf.keras.layers.BatchNormalization(name=bn_name)(x, training=train_bn)
    x = tf.keras.layers.Activation('relu', name=act_name)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name=pool_name)(x)
    return x


def backbone_protonet(x, filters=64, train_bn=False):
    """
    Builds a backbone of ProtoNet.

    Parameters
    ----------
    x : tf tensor, [batch_size, height, width, channels]
        An input tensor.
    filters : integer
        The number of filters/units in all conv layers. The default is 64.
    train_bn : boolean, optional
        Whether one should normalize the layer input by the mean and variance 
        over the current batch. The default is False, i.e., use the moving
        average of mean and variance to normalize the layer input.

    Returns
    -------
    x : tf tensor, [batch_size, height//2**4, width//2**4, filters] 
        The output tensor.

    """
    for i in range(4):
        x = conv_block(x, filters, str(i), train_bn)
    return x


def compute_shape(image_shape):
    """
    Computes output shape of the default ProtoNet.

    Parameters
    ----------
    image_shape : tuple
        The input image shape.

    Returns
    -------
    tuple
        The output shape [height//2**4, width//2**4, filters] .

    """
    assert len(image_shape) in [2, 3]
    if len(image_shape) == 2:
        image_shape = image_shape + (1,)
    inputs = tf.keras.Input(shape=image_shape)
    outputs = backbone_protonet(inputs, filters=64, train_bn=False)
    return outputs.shape[1:]

