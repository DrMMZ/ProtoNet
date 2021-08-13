"""
@author: Ming Ming Zhang, mmzhangist@gmail.com

Prototypical Networks
"""

import conv4, resnet
import tensorflow as tf


class GatherLayer(tf.keras.layers.Layer):
    """
    Defines a gathering layer as a subclass of TF layer.
    
    """
    def __init__(self, indices, axis, name=None):
        super(GatherLayer, self).__init__(name=name)
        self.indices = indices
        self.axis = axis
        
    def call(self, x):
        return tf.gather(x, indices=self.indices, axis=self.axis)
    

class CELoss(tf.keras.layers.Layer):
    """
    Defines a cross-entropy loss layer as a subclass of TF layer.
    
    """
    def __init__(self, name=None):
        super().__init__(name=name)
        
    def call(self, inputs):
        y_true, y_pred = inputs
        loss = tf.keras.backend.sparse_categorical_crossentropy(y_true, y_pred)
        loss = tf.math.reduce_mean(loss)
        return loss
    

class AccMetric(tf.keras.layers.Layer):
    """
    Defines an accuracy layer as a subclass of TF layer.
    
    """
    def __init__(self, name=None):
        super().__init__(name=name)
        
    def call(self, inputs):
        y_true, y_pred = inputs
        accs = tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        acc = tf.math.reduce_mean(accs)
        return acc


def protonet(
        mode, 
        N_C, 
        N_S, 
        image_shape, 
        train_bn=True, 
        N_Q=None, 
        N_T=None, 
        backbone='conv4'):
    """
    Builds a ProtoNet.

    Parameters
    ----------
    mode : string
        Either in 'training' or 'inference'.
    N_C : integer
        The number of classes/ways to sample.
    N_S : integer
        The number of support examples per class.
    image_shape : tuple
        The input image shape without batch size, i.e., (h, w, 3) for RGB or 
        (h, w, 1) for gray.
    train_bn : boolean, optional
        Whether one should normalize the layer input by the mean and variance 
        over the current batch. The default is False, i.e., use the moving
        average of mean and variance to normalize the layer input.
    N_Q : integer, optional
        The number of query examples per class in 'training' mode. The default 
        is None.
    N_T : integer, optional
        The number of test examples in 'inference' mode. The default is None.
    backbone : string, optional
        The backbone in {'conv4', 'resnet'}. The default is 'conv4'.

    Returns
    -------
    tf keras model
        The ProtoNet.

    """
    assert mode in ['training', 'inference']
    assert backbone in ['conv4', 'resnet']
    
    # input support images, [N_C * N_S, image_shape]
    S_images = tf.keras.Input(
            shape=image_shape, 
            # batch_size=N_C * N_S,
            name='S_images',
            dtype=tf.float32)
    
    # input query images [N_C * N_Q, image_shape] and class ids [N_C * N_Q, ]
    if mode == 'training':
        assert N_Q is not None, 'Require to specify N_Q.'
        # query images
        Q_images = tf.keras.Input(
            shape=image_shape, 
            # batch_size=N_C * N_Q,
            name='Q_images',
            dtype=tf.float32)
        # query class ids
        Q_class_ids = tf.keras.Input(
            shape=(), 
            # batch_size=N_C * N_Q,
            name='Q_class_ids',
            dtype=tf.int32)
        
    # input test images [N_T, image_shape]    
    else:
        assert train_bn == False, 'Require train_bn=False.'
        assert N_T is not None, 'Require to specify N_T.'
        # test images
        T_images = tf.keras.Input(
                shape=image_shape, 
                # batch_size=N_T,
                name='T_images')
            
    # concatenate input images for next computation
    if mode == 'training':
        # support and query images, [N_C*N_S + N_C*N_Q, image_shape]
        images = tf.keras.layers.Concatenate(axis=0)([S_images, Q_images])
    else:
        # support and test images, [N_C*N_S + N_T, image_shape]
        images = tf.keras.layers.Concatenate(axis=0)([S_images, T_images])
        
    # backbone in ['conv4', 'resnet']
    if backbone == 'conv4':
        # [(N_C*N_S + N_C*N_Q) or (N_C*N_S + N_T), fmap_h, fmap_w, 64]
        # conv4 with 64 filters
        x_images = conv4.backbone_protonet(
            images, 
            filters=64, 
            train_bn=train_bn)
        # flatten, [(N_C*N_S + N_C*N_Q) or (N_C*N_S + N_T), num_features], 
        # where num_features = fmap_h * fmap_w * 64
        x_images = tf.keras.layers.Flatten(name='flatten')(x_images)
    else:
        # [(N_C*N_S + N_C*N_Q) or (N_C*N_S + N_T), fmap_h, fmap_w, 1024]
        # resnet50 with stage5=False
        _, _, _, x_images, _ = resnet.backbone_resnet(
            images, 
            architecture='resnet50', 
            stage5=False, 
            train_bn=train_bn)
        # global average pooling, [(N_C*N_S + N_C*N_Q) or (N_C*N_S + N_T), 1024]
        x_images = tf.keras.layers.GlobalAveragePooling2D(name='pool')(x_images)
        # dense layer, [(N_C*N_S + N_C*N_Q) or (N_C*N_S + N_T), 1024]
        x_images = tf.keras.layers.Dense(units=1024, name='dense')(x_images)
        
    # support, [N_C*N_S, num_features/1024]
    x_S = GatherLayer(
        indices=tf.range(N_C * N_S), 
        axis=0,
        name='gather_support')(x_images)
    if mode == 'training':
        # query, [N_C*N_Q, num_features/1024]
        x_Q = GatherLayer(
            indices=tf.range(start=N_C * N_S, limit=N_C * N_S + N_C * N_Q), 
            axis=0,
            name='gather_query')(x_images)
    else:
        # test, [N_T, num_features/1024]
        x_T = GatherLayer(
            indices=tf.range(start=N_C * N_S, limit=N_C * N_S + N_T), 
            axis=0,
            name='gather_test')(x_images)
    
    # get prototype from x_S, [N_C, num_features/1024]
    def prototype(x):
        # reshape x, [N_C, N_S, num_features/1024]
        x = tf.reshape(x, shape=(N_C, N_S, -1))
        # prototype, [N_C, num_features/1024]
        p = tf.reduce_mean(x, axis=1)
        return p
    prototype_layer = tf.keras.layers.Lambda(
        prototype, 
        name='prototype')
    p_S = prototype_layer(x_S)
    
    # euclidean distance from query/test images to prototypes
    def euclidean(inputs):
        # x [N_C * N_Q or N_T, num_features/1024], y [N_C, num_features/1024]
        x, y = inputs
        batch_size = tf.shape(x)[0]
        num_features = tf.shape(x)[-1]
        # reshape x, [N_C * N_Q, N_C, num_features/1024]
        x = tf.expand_dims(x, axis=1)
        x = tf.broadcast_to(x, [batch_size, N_C, num_features])
        # euclidean distance, [N_C * N_Q, N_C]
        d = tf.norm(x - y, axis=-1)
        return d
    distance_layer = tf.keras.layers.Lambda(
        euclidean, 
        name='euclidean')
    if mode == 'training':
        # distance between x_Q and p_S, [N_C * N_Q, N_C]
        d = distance_layer([x_Q, p_S])
        # probs, [N_C * N_Q, N_C]
        scores = tf.keras.layers.Lambda(lambda x: -1 * x, name='scores')(d)
        probs = tf.keras.layers.Softmax(name='probs')(scores)
        # loss, scalar
        loss = CELoss(name='loss')([Q_class_ids, probs])
        # accuracy, scalar
        acc = AccMetric(name='acc')([Q_class_ids, probs])
        inputs = [S_images, Q_images, Q_class_ids]
        outputs = [d, probs, loss, acc]
    else:
        # distance between x_T and p_S, [N_T, N_C]
        d = distance_layer([x_T, p_S])
        # probs, [N_T, N_C]
        scores = tf.keras.layers.Lambda(lambda x: -1 * x, name='scores')(d)
        probs = tf.keras.layers.Softmax(name='probs')(scores)
        inputs = [S_images, T_images]
        outputs = [d, probs]
        
    with tf.device('/cpu:0'):
        model = tf.keras.Model(inputs, outputs, name='ProtoNet')
        return model

