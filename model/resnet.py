"""
Residual Networks (ResNet)
"""

# adapted from 
# https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py

import tensorflow as tf


def identity_block(
        input_tensor, 
        filters, 
        stage, 
        block, 
        train_bn=False
        ):
    """
    Builds an identity shortcut in a bottleneck building block of a ResNet.

    Parameters
    ----------
    input_tensor : tf tensor, [batch_size, height, width, channels]
        An input tensor.
    filters : list, positive integers
        The number of filters in 3 conv layers at the main path, where
        last number is equal to input_tensor's channels.
    stage : integer
        A number in [2,5] used for generating layer names.
    block : string
        A lowercase letter, used for generating layer names.
    train_bn : boolean, optional
        Whether one should normalize the layer input by the mean and variance 
        over the current batch. The default is False, i.e., use the moving
        average of mean and variance to normalize the layer input.

    Returns
    -------
    output_tensor : tf tensor, [batch_size, height, width, channels]
        The output tensor same shape as input_tensor.

    """
    num_filters_1, num_filters_2, num_filters_3 = filters
    conv_prefix = 'res' + str(stage) + block + '_branch'
    bn_prefix = 'bn' + str(stage) + block + '_branch'
    
    x = tf.keras.layers.Conv2D(
        num_filters_1, (1,1), name=conv_prefix + '2a')(input_tensor)
    x = tf.keras.layers.BatchNormalization(
        name=bn_prefix + '2a')(x, training=train_bn)
    x = tf.keras.layers.Activation('relu')(x)
    
    x = tf.keras.layers.Conv2D(
        num_filters_2, (3,3), padding='same', name=conv_prefix + '2b')(x)
    x = tf.keras.layers.BatchNormalization(
        name=bn_prefix + '2b')(x, training=train_bn)
    x = tf.keras.layers.Activation('relu')(x)
    
    x = tf.keras.layers.Conv2D(
        num_filters_3, (1,1), name=conv_prefix + '2c')(x)
    x = tf.keras.layers.BatchNormalization(
        name=bn_prefix + '2c')(x, training=train_bn)
    
    x = tf.keras.layers.Add()([input_tensor, x])
    output_tensor = tf.keras.layers.Activation(
        'relu', name='res' + str(stage) + block + '_out')(x)
    return output_tensor


def conv_block(
        input_tensor, 
        filters, 
        stage, 
        block, 
        strides=(2, 2), 
        train_bn=False
        ):
    """
    Builds a projection shortcut in a bottleneck block of a ResNet.

    Parameters
    ----------
    input_tensor : tf tensor, [batch_size, height, width, channels]
        An input tensor.
    filters : list, positive integers
        The number of filters in 3 conv layers at the main path.
    stage : integer
        A number in [2,5] used for generating layer names.
    block : string
        A lowercase letter, used for generating layer names.
    strides : tuple, integers, optional
        The conv layer strides. The default is (2, 2).
    train_bn : boolean, optional
        Whether one should normalize the layer input by the mean and variance 
        over the current batch. The default is False, i.e., use the moving
        average of mean and variance to normalize the layer input.

    Returns
    -------
    output_tensor : tf tensor 
        [batch_size, height//strides, width//strides, num_filters_3] where 
        num_filters_3 is the last number in filters, the output tensor.

    """
    num_filters_1, num_filters_2, num_filters_3 = filters
    conv_prefix = 'res' + str(stage) + block + '_branch'
    bn_prefix = 'bn' + str(stage) + block + '_branch'
    
    x = tf.keras.layers.Conv2D(
        num_filters_1, (1,1), strides, name=conv_prefix + '2a')(input_tensor)
    x = tf.keras.layers.BatchNormalization(
        name=bn_prefix + '2a')(x, training=train_bn)
    x = tf.keras.layers.Activation('relu')(x)
    
    x = tf.keras.layers.Conv2D(
        num_filters_2, (3,3), padding='same', name=conv_prefix + '2b')(x)
    x = tf.keras.layers.BatchNormalization(
        name=bn_prefix + '2b')(x, training=train_bn)
    x = tf.keras.layers.Activation('relu')(x)
    
    x = tf.keras.layers.Conv2D(
        num_filters_3, (1,1), name=conv_prefix + '2c')(x)
    x = tf.keras.layers.BatchNormalization(
        name=bn_prefix + '2c')(x, training=train_bn)
    
    shortcut = tf.keras.layers.Conv2D(
        num_filters_3, (1,1), strides, name=conv_prefix + '1')(input_tensor)
    shortcut = tf.keras.layers.BatchNormalization(
        name=bn_prefix + '1')(shortcut, training=train_bn)
    
    x = tf.keras.layers.Add()([shortcut, x])
    output_tensor = tf.keras.layers.Activation(
        'relu', name='res' + str(stage) + block + '_out')(x)
    return output_tensor


def backbone_resnet(
        input_image, 
        architecture='resnet50', 
        stage5=False, 
        train_bn=False):
    """
    Builds a backbone ResNet.

    Parameters
    ----------
    input_image : tf tensor, [batch_size, height, width, channels]
        An input tensor.
    architecture : string
        The ResNet architecture in {'resnet50', 'resnet101'}.
    stage5 : boolean, optional
        Whether create stage5 of network. The default is True.
    train_bn : boolean, optional
        Whether one should normalize the layer input by the mean and variance 
        over the current batch. The default is False, i.e., use the moving
        average of mean and variance to normalize the layer input.

    Returns
    -------
    outputs : list
        Feature maps at each stage.

    """
    assert architecture in ['resnet50', 'resnet101'], \
        'Only support ResNet50\101'
    
    # stage 1
    x = tf.keras.layers.ZeroPadding2D((3,3))(input_image)
    x = tf.keras.layers.Conv2D(64, (7,7), (2,2), name='conv1')(x)
    x = tf.keras.layers.BatchNormalization(name='bn_conv1')(x, training=train_bn)
    x = tf.keras.layers.Activation('relu')(x)
    C1 = x = tf.keras.layers.MaxPooling2D((3,3), (2,2), padding='same')(x)
    
    # stage 2
    x = conv_block(
        x, [64,64,256], stage=2, block='a', strides=(1,1), train_bn=train_bn)
    x = identity_block(x, [64,64,256], stage=2, block='b', train_bn=train_bn)
    C2 = x = identity_block(
        x, [64,64,256], stage=2, block='c', train_bn=train_bn)
    
    # stage 3
    x = conv_block(x, [128,128,512], stage=3, block='a', train_bn=train_bn)
    x = identity_block(x, [128,128,512], stage=3, block='b', train_bn=train_bn)
    x = identity_block(x, [128,128,512], stage=3, block='c', train_bn=train_bn)
    C3 = x = identity_block(
        x, [128,128,512], stage=3, block='d', train_bn=train_bn)
    
    # stage 4
    x = conv_block(x, [256,256,1024], stage=4, block='a', train_bn=train_bn)
    num_blocks = {'resnet50':5, 'resnet101':22}[architecture]
    for i in range(num_blocks):
        x = identity_block(
            x, [256,256,1024], stage=4, block=chr(98+i), train_bn=train_bn)
    C4 = x
    
    # stage 5
    if stage5:
        x = conv_block(x, [512,512,2048], stage=5, block='a', train_bn=train_bn)
        x = identity_block(
            x, [512,512,2048], stage=5, block='b', train_bn=train_bn)
        C5 = x = identity_block(
            x, [512,512,2048], stage=5, block='c', train_bn=train_bn)
    else:
        C5 = None
        
    return [C1, C2, C3, C4, C5]


def compute_shape(image_shape):
    """
    Computes each stage's output shape.

    Parameters
    ----------
    image_shape : tuple
        The input image shape.

    Returns
    -------
    tuple
        Each is the corresponding stage's output shape without batch size.

    """
    assert len(image_shape) in [2, 3]
    if len(image_shape) == 2:
        image_shape = image_shape + (1,)
    inputs = tf.keras.Input(shape=image_shape)
    C1, C2, C3, C4, C5 = backbone_resnet(
        inputs, architecture='resnet50', stage5=True, train_bn=False)
    return C1.shape[1:], C2.shape[1:], C3.shape[1:], C4.shape[1:], C5.shape[1:]