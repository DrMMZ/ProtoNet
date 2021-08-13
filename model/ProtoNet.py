"""
@author: Ming Ming Zhang, mmzhangist@gmail.com

Prototypical Networks OOP
"""

import protonet_model
import os, datetime, time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score


class ProtoNet(object):
    """
    Defines a class based on ProtoNet, including training (can use synchronized 
    multi-gpu training) and inference.
    
    """
    
    def __init__(self, mode, config):
        """
        A constructor.

        Parameters
        ----------
        mode : string
            The mode of building a retinanet in {'training', 'inference'}.
        config : class
            A custom configuration, see config.Config().

        Returns
        -------
        None.

        """
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        
        if mode == 'inference':
            self.model = self.build_protonet(mode, config)          
            if config.checkpoint_path is not None:
                print('\nLoading checkpoint:\n%s\n'%config.checkpoint_path)
                self.model.load_weights(
                    config.checkpoint_path, 
                    by_name=False)
        
        
    def build_protonet(self, mode, config):
        """
        Builds a ProtoNet.

        Parameters
        ----------
        mode : string
            The mode of building a retinanet in {'training', 'inference'}.
        config : class
            A custom configuration, see config.Config().

        Returns
        -------
        model : tf keras model
            A protonet based on the given config.

        """
        # note that config.N_C is config.batch_size_per_gpu
        model = protonet_model.protonet(
            mode=mode, 
            N_C=config.N_C, 
            N_S=config.N_S, 
            image_shape=config.image_shape, 
            train_bn=config.train_bn, 
            N_Q=config.N_Q, 
            N_T=config.N_T,
            backbone=config.backbone)
        return model
    
    
    def compile_model(
            self, 
            model, 
            lr, 
            momentum, 
            beta_2, 
            l2, 
            metric_names=['loss', 'acc']
            ):
        """
        Add Adam optimizer, loss and L2-regularization to the model.

        Parameters
        ----------
        model : tf keras model
            The built ProtoNet.
        lr : float
            A learning rate.
        momentum : float
            A scalar in Adam controlling moving average of the gradients decay.
        beta_2 : float
            A scalar in Adam controlling moving average of the squared gradients 
            decay.
        l2 : float
            A scalar in L2-regularization controlling the strength of 
            regularization.
        metric_names : list, optional
            The name(s) of metric function(s) in the model. The default is 
            ['loss', 'acc'], i.e., cross-entropy (classification) loss and 
            accuracy defined in protonet.CELoss() and protonet.AccMetric(),
            respectively.

        Returns
        -------
        None.

        """
        # optimizer
        optimizer = tf.keras.optimizers.Adam(
            lr=lr, 
            beta_1=momentum, 
            beta_2=beta_2,
            epsilon=1e-7)
                
        # loss and metrics
        for name in metric_names:
            layer = model.get_layer(name)
            output = layer.output
            if name == 'loss':
                model.add_loss(output)
            model.add_metric(output, name=name)
                    
        # l2-regularization, exclude batch norm weights
        reg_losses = []
        for w in model.trainable_weights:
            if 'gamma' not in w.name and 'beta' not in w.name:
                reg_losses.append(
                    tf.math.divide(
                        tf.keras.regularizers.L2(l2)(w),
                        tf.cast(tf.size(w), w.dtype)))
        model.add_loss(lambda: tf.math.add_n(reg_losses))
        
        # compile the model            
        model.compile(
            optimizer=optimizer, 
            loss=[None] * len(model.outputs))
        
        
    def train(self, 
              train_generator, 
              val_generator=None, 
              plot_training=True
              ): 
        """
        Trains the built ProtoNet.

        Parameters
        ----------
        train_generator : python generator
            Described in data.data_gen().
        val_generator : python generator, optional
            Described in data.data_gen(). The default is None.
        plot_training : boolean, optional
            Whether to plot the learning curves. The default is True.

        Returns
        -------
        The trained model, training log and plot if needed.

        """
        assert self.mode == 'training', \
            'Need to create an instance in training mode.'
            
        if self.config.num_gpus > 1 and \
            len(tf.config.list_physical_devices('GPU')) > 1:
                strategy = tf.distribute.MirroredStrategy(
                    cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()
                    )
        else:
            strategy = tf.distribute.get_strategy() 
            
        with strategy.scope():
            self.model = self.build_protonet(self.mode, self.config)
            
            if self.config.checkpoint_path is not None:
                print('\nLoading checkpoint:\n%s\n' \
                      % self.config.checkpoint_path)
                self.model.load_weights(
                    self.config.checkpoint_path, by_name=False)
                
            if self.config.resnet_weights_path is not None:
                print('\nLoading resnet:\n%s\n' \
                      % self.config.resnet_weights_path)
                self.model.load_weights(
                    self.config.resnet_weights_path, by_name=True)
                # freeze some layers
                if self.config.freeze_layers:
                    assert self.config.train_bn == False, \
                        'Require train_bn=False.'
                    for i in range(len(self.model.layers)):
                        layer = self.model.layers[i]
                        if layer.name == 'dense':
                            assert self.model.layers[i-1].name == 'pool'
                            break              
                        layer.trainable = False
        
            self.compile_model(
                self.model, 
                self.config.lr, 
                self.config.momentum, 
                self.config.beta_2,
                self.config.l2)
            # assign a learning rate after loading a checkpoint; otherwise it 
            # will continue on the last learning rate in the checkpoint
            self.model.optimizer.lr.assign(self.config.lr)
            print('\nlearning rate:', self.model.optimizer.lr.numpy(), '\n')
        
        # callbacks, including CSVLogger, ModelCheckpoint, ReduceLROnPlateau,
        # and EarlyStopping
        callbacks = []
        ROOT_DIR = os.getcwd()
        log_dir = os.path.join(ROOT_DIR, 'checkpoints')
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        checkpoint_dir = os.path.join(log_dir, current_time)
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        if self.config.save_weights:
            self.checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint')
            cp_callback = tf.keras.callbacks.ModelCheckpoint(
                self.checkpoint_path, 
                save_weights_only=True)
            callbacks.append(cp_callback)
        if self.config.reduce_lr:
            if val_generator is not None:
                reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', 
                    factor=0.1, 
                    patience=self.config.reduce_lr_patience)
            else:
                reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='loss', 
                    factor=0.1, 
                    patience=self.config.reduce_lr_patience)
            callbacks.append(reduce_lr_callback)
        if self.config.early_stopping:
            if val_generator is not None:
                early_stopping_callback = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', 
                    patience=self.config.early_stopping_patience, 
                    restore_best_weights=True)
            else:
                early_stopping_callback = tf.keras.callbacks.EarlyStopping(
                    monitor='loss', 
                    patience=self.config.early_stopping_patience, 
                    restore_best_weights=True)
            callbacks.append(early_stopping_callback)
        log_filename = os.path.join(checkpoint_dir, '%s.csv' % current_time)
        log_callback = tf.keras.callbacks.CSVLogger(
            log_filename, 
            append=False)
        callbacks.append(log_callback)
        
        # train
        if val_generator is not None:
            history = self.model.fit(
                train_generator, 
                epochs=self.config.epochs, 
                steps_per_epoch=self.config.steps_per_epoch, 
                callbacks=callbacks, 
                validation_data=val_generator, 
                validation_steps=self.config.validation_steps,
                validation_freq=self.config.validation_freq)
        else:
            history = self.model.fit(
                train_generator, 
                epochs=self.config.epochs, 
                steps_per_epoch=self.config.steps_per_epoch, 
                callbacks=callbacks)
        
        # learning curves, saved to checkpoint_dir
        if plot_training:
            d = history.history
            plt.figure(figsize=(10, 10))
            for i in range(2):
                plt.subplot(2,1,i+1)
                plt.plot(d[list(d.keys())[0+i]], label='train')
                if val_generator is not None:
                    plt.plot(d[list(d.keys())[2+i]], label='val')
                plt.title(list(d.keys())[0+i])
                plt.legend()
            plt.savefig(os.path.join(checkpoint_dir, '%s.png' % current_time))
            plt.show()
            
            
    def predict(self, S_images, T_images, dist_cf_threshold=None):
        """
        Given support images, classifies test images. 

        Parameters
        ----------
        S_images : tf tensor/numpy array, [N_C * N_S, image_shape]
            A given support images.
        T_images : tf tensor/numpy array, [N_T, image_shape]
            A given test images.
        dist_cf_threshold : float, optional
            A threshold to classify T_images class ids, i.e., if the distance 
            between a test image and support images is greater than the 
            threshold, then the test class id is denoted by -1, indicating it 
            is not in support class ids; otherwise, it is the corresponding 
            support class id. The default is None.

        Returns
        -------
        class_ids : numpy array, [N_T, ]
            The predicted class ids of T_images in {0,...,N_C-1} if 
            dist_cf_threshold=None; {-1,0,...,N_C-1} otherwise.
        d_min : numpy array, [N_T, ]
            The minimum distance between every test image and support images.

        """
        assert self.mode == 'inference', \
            'Need to create an instance in inference mode.'
        assert np.mean(S_images) < 1 and np.mean(T_images) < 1, \
            'Require inputs to be divided by 255.'

        # distance d and probs, [N_T, N_C]
        d, probs = self.model([S_images, T_images])
        # predicted class ids, [N_T, ]
        class_ids = np.argmin(d.numpy(), axis=1)
        # minimum distance, [N_T, ]
        d_min = np.amin(d.numpy(), axis=1)
        
        if dist_cf_threshold is not None:
            # indices for distances in d > dist_cf_threshold
            idxes = np.where(d_min > dist_cf_threshold)[0]
            # filter predicted class ids by setting to -1, [N_T, ]
            class_ids[idxes] = -1
        
        return class_ids, d_min
    
    
    def evaluate(
            self, 
            S_images, 
            T_images, 
            T_class_ids, 
            dist_cf_threshold=None,
            visualize=False
            ):
        """
        Given support images, evaluates test images.

        Parameters
        ----------
        S_images : tf tensor/numpy array, [N_C * N_S, image_shape]
            A given support images.
        T_images : tf tensor/numpy array, [N_T, image_shape]
            A given test images.
        T_class_ids : tf tensor/numpy array, [N_T, ]
            The ground-truth class ids of T_images in {-1,0,...,N_C-1} where
            -1 indicates the class is not in support classes.
        dist_cf_threshold : float, optional
            Same as above. The default is None.
        visualize : boolean, optional
            Whether to visualize learning and prediction. The default is False.

        Returns
        -------
        acc : float
            The accuracy.
        f1 : float
            The F1-score.
        d_min : numpy array, [N_T, ]
            Same as above.

        """
        t1 = time.time()
        pred_T_class_ids, d_min = self.predict(
            S_images, 
            T_images, 
            dist_cf_threshold)
        t2 = time.time()
        acc = np.mean(pred_T_class_ids == T_class_ids)
        f1 = f1_score(T_class_ids, pred_T_class_ids, average='macro')
        
        if visualize:
            print('time: %fs\n' %(t2-t1))
            print('metrics: acc=%f, f1=%f'%(acc, f1))
            plt.figure(figsize=(12,5))
            plt.suptitle('Learning', weight='bold', fontsize=18)
            for i in range(self.config.N_S):
                plt.subplot(1, self.config.N_S, i+1)
                tmp = S_images[i]
                plt.imshow(tmp)
                plt.axis('off')
            plt.show()
            print('\n')      
            plt.figure(figsize=(12,8))
            plt.suptitle('Recognizing', weight='bold', fontsize=18)
            for i in range(len(T_images)):
                plt.subplot(3, int(np.ceil(len(T_images)/3)), i+1)
                tmp = T_images[i]
                plt.imshow(tmp)
                class_name = T_class_ids[i]
                if i == np.argmin(d_min) and class_name != -1:
                    plt.title('match', color='green', weight='bold')
                elif i == np.argmin(d_min) and class_name == -1:
                    plt.title('predict', color='red', weight='bold')
                elif i != np.argmin(d_min) and class_name != -1:
                    plt.title('ground-truth', color='red', weight='bold')
                plt.axis('off')
            plt.show()
            
        return acc, f1, d_min
            
        