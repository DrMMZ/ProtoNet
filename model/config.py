"""
@author: Ming Ming Zhang, mmzhangist@gmail.com

Configurations
"""

import numpy as np


class Config(object):
    """
    Defines a custom configurations for ProtoNet.
    
    """
    
    name = None 
    
    ################
    # The model
    ################
    N_C = None # number of classes per episode
    N_S = None # number of support examples per class per episode
    N_Q = None # number of query examples per class per episode
    # number of test examples not in queries for all classes, 
    # i.e., N_T = N_C * N_Q + N_T_Q
    N_T_Q = None 
    train_bn = True # train batch norm layers?
    backbone = 'conv4' # conv4 or resnet?
    
    ################
    # Training
    ################
    num_gpus = 1 # used in multi-GPU training or inferencing
    checkpoint_path = None # previous trained model weights path
    resnet_weights_path = None # previous trained resnet weights path
    freeze_layers = False # freeze resnet layers?
    lr = 1e-4 # learning rate
    momentum = 0.9 # Adam opt scalar for moving average of grads decay
    beta_2 = 0.999 # Adam opt scalar for moving average of squared grads decay
    l2 = 1e-4 # L2 regularization strength
    save_weights = True # save trained weights?
    epochs = 1 # epochs
    validation_freq = 1 # frequence to validate
    reduce_lr = False # apply ReduceLROnPlateau?
    reduce_lr_patience = 10 # num of epochs should run after best
    early_stopping = False # apply EarlyStopping?
    early_stopping_patience = 10 # num of epochs should run after best
    
    ################
    # Image
    ################
    img_mode = 'rgb' # image is RGB or gray
    shortest_side = None # shortest side after resized
    longest_side = None # longest side after resized
    resize_mode = 'pad_square' # in {'crop', 'pad_square', 'none'}
    
    ################
    # Dataset
    ################
    num_train_images = None # number of training images
    num_val_images = None # number of validation images 
    num_test_images = None # number of test images
    
    
    def __init__(self):
        # image shape
        if self.resize_mode == 'crop':
            if self.img_mode == 'rgb':
                self.image_shape = (self.shortest_side, self.shortest_side, 3)
            else:
                self.image_shape = (self.shortest_side, self.shortest_side, 1)
        elif self.resize_mode == 'pad_square':
            if self.img_mode == 'rgb':
                self.image_shape = (self.longest_side, self.longest_side, 3)
            else:
                self.image_shape = (self.longest_side, self.longest_side, 1)
        else:
            # need to manually adjust if resize_mode = 'none'
            self.image_shape = (218, 178, 3)
            
        # batch size
        self.batch_size_per_gpu = self.N_C
        self.batch_size_global = self.batch_size_per_gpu * self.num_gpus
        
        # number of test examples, N_T
        if self.N_T_Q is not None:
            self.N_T = self.batch_size_global * self.N_Q + self.N_T_Q
            self.num_images_per_episode = \
                self.batch_size_global * self.N_S + self.N_T
        else:
            self.num_images_per_episode = \
                self.batch_size_global * (self.N_S + self.N_Q)
        assert self.num_images_per_episode < self.num_train_images, \
                "Training set doesn't have enough images to sample"
        if self.num_val_images is not None:
            assert self.num_images_per_episode < self.num_val_images, \
                    "Validation set doesn't have enough images to sample"
        
        # number of training, val and test steps per epoch
        self.steps_per_epoch = int(np.ceil(
            self.num_train_images / self.num_images_per_episode))
        if self.num_val_images is not None:
            self.validation_steps = int(np.ceil(
                self.num_val_images / self.num_images_per_episode))
        if self.num_test_images is not None:
            self.steps_per_epoch = int(np.ceil(
                self.num_test_images / self.num_images_per_episode))
            self.validation_steps = None
            self.validation_freq = None
        
        
    def display(self):
        print('----------Configurations----------\n')
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))