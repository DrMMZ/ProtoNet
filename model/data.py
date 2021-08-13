"""
@author: Ming Ming Zhang, mmzhangist@gmail.com

ProtoNet Data Generator
"""

import numpy as np
import concurrent.futures


def data_gen(
        dataset, 
        config,
        mode='training',
        shuffle=True):
    """
    Generates the input data asynchronously (multi-CPU).

    Parameters
    ----------
    dataset : class
        Described in utils.Dataset().
    config : class
        A custom configuration, see config.Config().
    mode : string, optional
        Data mode in {'training', 'inference', 'inspection'}. The default is 
        'training'.
    shuffle : boolean, optional
        Whether to shuffle the query examples in each episode. The default is 
        True.

    Yields
    ------
    data : dictionary in each mode where N_C = config.batch_size_global
        * 'training' : 
            - 'S_images' support examples [N_C * N_S, image_shape], 
            - 'Q_images' query examples [N_C * N_Q, image_shape], and 
            - 'Q_class_ids' query class ids [N_C * N_Q, ] in 
              {0, ..., config.batch_size_per_gpu - 1}.
        * 'inference' : 
            - 'S_images' support examples [N_C * N_S, image_shape], 
            - 'T_images' test examples [N_T, image_shape] where 
              N_T = N_C * N_Q + N_T_Q, and 
            - 'T_class_ids' test class ids [N_C * N_Q, ] in 
              {-1, 0, ..., config.batch_size_per_gpu - 1} where -1 indicates 
              the class is not in support classes.
        * 'inspection' : 
            - 'S_images' support examples [N_C * N_S, image_shape], 
            - 'S_class_ids' support class ids [N_C * N_S, ] in 
              {0, ..., config.batch_size_per_gpu - 1},
            - 'S_class_names' support class names [N_C * N_S, ],
            - 'Q_images' query examples [N_C * N_Q, image_shape],
            - 'Q_class_ids' query class ids [N_C * N_Q, ] in 
              {0, ..., config.batch_size_per_gpu - 1}, and
            - 'Q_class_names' query class names [N_C * N_Q, ],

    """
    assert mode in ['training', 'inference', 'inspection']
    assert config.batch_size_global < dataset.num_classes, \
        "Dataset doesn't have enough classes to sample."
    
    # note that num of classes N_C in each episode is config.batch_size_global,
    # i.e., maybe used in multi-GPU training
    N_C, N_S, N_Q = config.batch_size_global, config.N_S, config.N_Q
    class_ids = np.copy(dataset.class_ids)
    
    if mode == 'inference':
        N_T_Q = config.N_T_Q   

    # for each episode
    while True:
        some_class_ids = np.random.choice(class_ids, size=N_C, replace=False)
        
        if mode == 'inference':
            extra_class_ids = np.random.choice(
                np.setdiff1d(class_ids, some_class_ids), 
                size=N_T_Q)
        
        # image ids for support and query examples, [N_C * (N_S+N_Q), ] 
        SQ_image_ids = []
        for k in some_class_ids:
            # get image ids for the class k
            image_ids_k = dataset.class2images(k)
            assert len(image_ids_k) >= N_S + N_Q, \
                "Class %d, %s doesn't have enough images to sample." \
                    % (k, dataset.class_info[k]['name'])
            SQ_image_ids_k = np.random.choice(
                image_ids_k, 
                size=N_S+N_Q, 
                replace=False)
            SQ_image_ids.append(SQ_image_ids_k)
        SQ_image_ids = np.concatenate(SQ_image_ids, axis=0) 
        
        if mode == 'inference':
            extra_image_ids = []
            for k in extra_class_ids:
                image_ids_k = dataset.class2images(k)
                extra_image_ids_k = np.random.choice(image_ids_k, size=1)
                extra_image_ids.append(extra_image_ids_k)
            if len(extra_image_ids) > 0:
                extra_image_ids = np.concatenate(extra_image_ids, axis=0)
            else:
                extra_image_ids = np.array([], np.int32)
        
        # generate data (images, class ids & image names), [N_C * (N_S+N_Q), ...] 
        SQ_images = np.zeros(
            shape=(N_C * (N_S+N_Q), ) + config.image_shape, 
            dtype=np.float32)
        SQ_class_ids = np.zeros(
            shape=(N_C * (N_S+N_Q), ), 
            dtype=np.int32)
        SQ_class_names = np.empty((N_C * (N_S+N_Q),), dtype='object')
        
        if mode == 'inference':
            extra_images = np.zeros(
                shape=(N_T_Q, ) + config.image_shape, 
                dtype=np.float32)
            extra_class_ids = -1 * np.ones(
                shape=(N_T_Q, ), 
                dtype=np.int32)
            extra_class_names = np.empty((N_T_Q,), dtype='object')
        
        def fn(image_id, b):
            # get resized image and its class id
            image, class_id, cache = dataset.load_data(
                image_id, 
                img_mode=config.img_mode,
                shortest_side=config.shortest_side, 
                longest_side=config.longest_side, 
                #config.upscale_factor,
                resize_mode=config.resize_mode,
                verbose=0)
            # expand grayscale image dim
            if config.image_shape[-1] == 1:
                image = np.expand_dims(image, axis=-1)
            # add to batch
            SQ_images[b] = image / 255.0
            SQ_class_ids[b] = class_id
            SQ_class_names[b] = cache[1]
                
        def fn_extra(image_id, b):
            # get resized image and its class id
            image, class_id, cache = dataset.load_data(
                image_id, 
                img_mode=config.img_mode,
                shortest_side=config.shortest_side, 
                longest_side=config.longest_side, 
                #config.upscale_factor,
                resize_mode=config.resize_mode,
                verbose=0)
            # expand grayscale image dim
            if config.image_shape[-1] == 1:
                image = np.expand_dims(image, axis=-1)
            # add to batch
            extra_images[b] = image / 255.0
            extra_class_names[b] = cache[1]      
             
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for b in range(N_C * (N_S+N_Q)):
                executor.submit(fn, SQ_image_ids[b], b)
                
        if mode == 'inference':
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for b in range(N_T_Q):
                    executor.submit(fn_extra, extra_image_ids[b], b)
                
        # get support and query examples and class ids 
        tmp = SQ_images.reshape((N_C, N_S+N_Q,) + config.image_shape)
        S_images = tmp[:, :N_S] # [N_C, N_S, image_shape]
        Q_images = tmp[:, N_S:] # [N_C, N_Q, image_shape]
        tmp = SQ_class_ids.reshape((N_C, N_S+N_Q)) 
        S_class_ids = tmp[:, :N_S] # [N_C, N_S]
        Q_class_ids = tmp[:, N_S:] # [N_C, N_Q]
        tmp = SQ_class_names.reshape((N_C, N_S+N_Q)) 
        S_class_names = tmp[:, :N_S] # [N_C, N_S]
        Q_class_names = tmp[:, N_S:] # [N_C, N_Q]
        
        # reshape support and query examples and class ids
        S_images = np.reshape(
            S_images, 
            newshape=(-1,) + config.image_shape) # [N_C * N_S, image_shape]
        Q_images = np.reshape(
            Q_images, 
            newshape=(-1,) + config.image_shape) # [N_C * N_Q, image_shape]
        S_class_ids = np.reshape(S_class_ids, newshape=(-1,)) # [N_C * N_S]
        Q_class_ids = np.reshape(Q_class_ids, newshape=(-1,)) # [N_C * N_Q]
        S_class_names = np.reshape(S_class_names, newshape=(-1,)) # [N_C * N_S]
        Q_class_names = np.reshape(Q_class_names, newshape=(-1,)) # [N_C * N_Q]
        
        # shuffle query examples and class ids
        tmp1, tmp2, tmp3 = [], [], []
        if shuffle:
            # for each gpu, shuffle corresponding examples, class ids and names
            for i in range(config.num_gpus):
                start = i * config.batch_size_per_gpu * N_Q
                end = (i+1) * config.batch_size_per_gpu * N_Q 
                Q_images_per_gpu = Q_images[start:end]
                Q_class_ids_per_gpu = Q_class_ids[start:end]
                Q_class_names_per_gpu = Q_class_names[start:end]
                idxes = np.arange(len(Q_images_per_gpu))
                np.random.shuffle(idxes)
                Q_images_per_gpu = Q_images_per_gpu[idxes] # [N_C * N_Q, image_shape]
                Q_class_ids_per_gpu = Q_class_ids_per_gpu[idxes] # [N_C * N_Q]
                Q_class_names_per_gpu = Q_class_names_per_gpu[idxes] # [N_C * N_Q]
                tmp1.append(Q_images_per_gpu)
                tmp2.append(Q_class_ids_per_gpu)
                tmp3.append(Q_class_names_per_gpu)
            Q_images = np.concatenate(tmp1, axis=0)
            Q_class_ids = np.concatenate(tmp2, axis=0)
            Q_class_names = np.concatenate(tmp3, axis=0)
            
        if mode in ['training', 'inference']:
            # relabel query class ids
            Q_class_ids_relabeled = []
            # for each gpu, map Q_class_ids_per_gpu to the corresponding indices 
            # of S_class_ids_per_gpu
            for i in range(config.num_gpus):
                start = i * config.batch_size_per_gpu * N_S
                end = (i+1) * config.batch_size_per_gpu * N_S
                S_class_ids_per_gpu = S_class_ids[start:end]
                # get unique S_class_ids per gpu
                _, idxes = np.unique(S_class_ids_per_gpu, return_index=True)
                S_class_ids_unique = S_class_ids_per_gpu[np.sort(idxes)]
                # get labels per gpu
                for j in range(
                        i * config.batch_size_per_gpu * N_Q, 
                        (i+1) * config.batch_size_per_gpu * N_Q):
                    class_id = Q_class_ids[j]
                    new_label = np.where(S_class_ids_unique == class_id)[0]
                    Q_class_ids_relabeled.append(new_label)
            Q_class_ids_relabeled = np.concatenate(
                Q_class_ids_relabeled, 
                axis=0).astype(Q_class_ids.dtype)
            if mode == 'training':
                # SQ_images = np.concatenate((S_images, Q_images), axis=0)
                data = {
                    'S_images':S_images,
                    'Q_images':Q_images,
                    # 'SQ_images':SQ_images,
                    'Q_class_ids':Q_class_ids_relabeled,
                    # 'S_class_names':S_class_names,
                    # 'Q_class_names':Q_class_names
                    }
            else:
                # [N_T, image_shape], where N_T = N_C * N_Q + N_T_Q
                T_images = np.concatenate((Q_images, extra_images), axis=0)
                # [N_T, ]
                T_class_ids = np.concatenate(
                    (Q_class_ids_relabeled, extra_class_ids),
                    axis=0)
                # print('---T_images:', T_images.shape, T_images.dtype)
                # print('---T_class_ids:', T_class_ids.shape, T_class_ids.dtype)
                idxes = np.arange(len(T_images))
                np.random.shuffle(idxes)
                T_images = T_images[idxes]
                T_class_ids = T_class_ids[idxes]
                data = {
                    'S_images':S_images,
                    'T_images':T_images,
                    'T_class_ids':T_class_ids,
                    }
            yield data
            
        else:
            data = {
                'S_images':S_images,
                'S_class_ids':S_class_ids,
                'S_class_names':S_class_names,
                'Q_images':Q_images,
                'Q_class_ids':Q_class_ids,
                'Q_class_names':Q_class_names
                }
            yield data
            