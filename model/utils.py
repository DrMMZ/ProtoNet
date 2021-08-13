"""
@author: Ming Ming Zhang, mmzhangist@gmail.com

Utilities
"""

import time
import random
import numpy as np
import skimage.io, skimage.color, skimage.transform


############################################################
#  Image Preprocessing
############################################################

def load_image(image_path, mode):
    """
    Loads an image into RGB or gray.

    Parameters
    ----------
    image_path : string
        The image file path.

    Returns
    -------
    image : numpy array, [h, w, 3] for RGB or [h, w] for gray
        Loaded image.

    """
    image = skimage.io.imread(image_path)
    
    assert mode in ['gray', 'rgb']
    if mode == 'rgb':
        # convert to RGB if it is grayscale
        if image.ndim != 3:
            # print('%s was grayscale'%image_path)
            image = skimage.color.gray2rgb(image)     
        # convert to RGB if it is RGBA
        if image.shape[-1] == 4:
            # print('%s was RGBA'%image_path)
            image = image[..., :3] 
            
    else:
        # convert to grayscale if it is RGB
        if image.ndim != 2:
            # print('%s was RGB'%image_path)
            image = skimage.color.rgb2gray(image) 
        
    return image


def resize_image(
        image, 
        shortest_side=512, 
        longest_side=1024, 
        upscale_factor=1.0,
        mode='crop'
        ):
    """
    Resizes an image without changing the aspect ratio.

    Parameters
    ----------
    image : numpy array, [height, width, channels]
        The image needed to be resized.
    shortest_side : integer, optional
        The shortest side of the resized image. Note that it only applies to
        the 'crop' mode. The default is 512.
    longest_side : integer, optional
        The longest side of the resized image. Note that it only applies to the
        'pad_square' mode. The default is 1024.
    upscale_factor : float, optional
        The scale factor >= 1.0 to upscale the image. The default is 1.0.
    mode : string, optional
        The resizing method in {'none'', 'pad_square', 'crop}. The default is 
        'crop'.
        * 'none' : [height, width, channels], no resizing nor padding applied.
        * 'pad_square' : [longest_side, longest_side, channels], padded with 0 
        to keep the same aspect ratio.
        * 'crop' : [shorest_side, shortest_side, channels].

    Returns
    -------
    image : numpy array, [resized_h, resized_w, channels]
        The resized image depending on the mode with the same data type as the
        input image.
    window : tuple
        The corner coordinates (y1, x1, y2, x2) indicates the location of the
        resized image before padding.
    scale : float
        The scale factor from original to resized image, and used in the later
        resize_mask().
    padding : list
        The padding applied to the original image in the form of pixels
        [(top, bottom), (left, right), (0, 0)], and used in the later
        resize_mask().
    crop : tuple
        The coordinates (y1, x1, h, w) indicates the location of the
        cropped image over the resized image, and used in the later 
        resize_mask().

    """
    assert mode in ['none', 'pad_square', 'crop']
    
    image_dtype = image.dtype
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1.0
    if image.ndim == 3:
        padding = [(0,0), (0,0), (0,0)]
    else:
        padding = [(0,0), (0,0)]
    crop = None
    
    if mode == 'none':
        return image, window, scale, padding, crop
    
    # upscale for all modes, scale >= 1.0
    scale = max(1.0, shortest_side / min(h, w))
    if scale < upscale_factor:
        scale = upscale_factor
        
    # downscale only if mode == 'pad_square', scale < 1.0
    if mode == 'pad_square':
        max_side = max(h, w)
        if round(max_side * scale) > longest_side:
            scale = longest_side / max_side
            
    # resize image using bilinear
    if scale != 1.0:
        image = skimage.transform.resize(
            image, 
            output_shape=(round(h * scale), round(w * scale)),
            preserve_range=True, 
            anti_aliasing=False)
        
    if mode == 'pad_square':
        h, w = image.shape[:2]
        top_pad = (longest_side - h) // 2
        bottom_pad = longest_side - h - top_pad
        left_pad = (longest_side - w) // 2
        right_pad = longest_side - w - left_pad
        if image.ndim == 3:
            padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        else:
            padding = [(top_pad, bottom_pad), (left_pad, right_pad)]
        image = np.pad(image, padding)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
        
    else:
        h, w = image.shape[:2]
        # note that h (or w) - shortest_side maybe 0
        y = random.randint(0, (h - shortest_side))
        x = random.randint(0, (w - shortest_side))
        crop = (y, x, shortest_side, shortest_side)
        image = image[y:y + shortest_side, x:x + shortest_side]
        window = (0, 0, shortest_side, shortest_side)
        
    return image.astype(image_dtype), window, scale, padding, crop
    

############################################################
#  Dataset
############################################################

class Dataset(object):
    """
    Defines a dataset class.
    
    """
    
    def __init__(self):
        # image_info, list of dictionaries 
        # {'id':image_id, 'path':path, 'class_name':class_name, ...}
        self.image_info = []
        
        # class_info, list of dictionaries
        # {'id':class_id, 'name':class_name}
        self.class_info = []
    
        
    def add_image(self, image_id, path, class_name, **kwargs):
        """
        Adds image information to the dataset.
 
        Parameters
        ----------
        image_id : string/integer
            The image ID.
        path : string
            The image path.
        class_name : string
            The corresponding class name.
        **kwargs : string/integer
            Additional information, e.g., image's height, width and so on.
 
        Returns
        -------
        None.
 
        """
        image_info = {'id':image_id, 'path':path, 'class_name':class_name, }
        image_info.update(kwargs)
        self.image_info.append(image_info)
       
       
    def add_class(self, class_id, class_name):
        """
        Adds class information to the dataset.

        Parameters
        ----------
        class_id : integer
            The class ID.
        class_name : string
            The corresponding class name. Augmented class names have the forms:
                * name_rot_n : rotate n degree where n in [90,180,270]
                * name_flip_lr : flip horizontally
                * name_flip_ud : flip vertically

        Returns
        -------
        None.

        """       
        for info in self.class_info:
            if info['id'] == class_id:
                # skip if class_id already existed
                return
        self.class_info.append({'id':class_id, 'name':class_name})
         
         
    def prepare(self):
        """
        Prepares the dataset and needs to be called before using it.

        Returns
        -------
        None.

        """
        # info from class_info and image_info
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [
            a_class_info['name'] for a_class_info in self.class_info]
        self.num_images = len(self.image_info)
        self.image_ids = np.arange(self.num_images)
        
        
    def get_ds_info(self):
        """
        Gets the dataset information.

        Returns
        -------
        class_names : list
            Each element is a name corresponding to the class id.

        """
        print('There are %d images and %d classes.' \
              % (self.num_images, self.num_classes))
        return self.class_names
        
        
    def get_img_info(self, image_id):
        """
        Gets an image information.

        Parameters
        ----------
        image_id : integer
            An image id from image_ids.

        Returns
        -------
        image_info : dictionary
            The image information with keys 'id', 'path' and so on; see 
            add_image().

        """
        return self.image_info[image_id]
    
    
    def image_name2id(self, image_id_str):
        """
        Gets the image id from the name of image.

        Parameters
        ----------
        image_id_str : string
            The name of image, image_info['id'].

        Returns
        -------
        image_id : integer
            The corresponding image id.

        """
        for image_id in self.image_ids:
            if self.get_img_info(image_id)['id'] == image_id_str:
                return image_id
            
    
    def class_name2id(self, class_name):
        """
        Gets the class id from the name of class.

        Parameters
        ----------
        class_name : string
            The given class name.

        Returns
        -------
        integer
            The corresponding class id.

        """
        for class_id in self.class_ids:
            if class_name == self.class_info[class_id]['name']:
                return self.class_info[class_id]['id']
      
            
    def class2images(self, class_id):
        """
        Get image ids from the class id. It needs to be re-defined in a 
        subclass; otherwise it won't be efficient.

        Parameters
        ----------
        class_id : integer
            The given class id.

        Returns
        -------
        image_ids : list
            The set of image ids for the given class id.

        """
        image_ids = []
        for image_id in self.image_ids:
            class_name = self.get_img_info(image_id)['class_name']
            if self.class_name2id(class_name) == class_id:
                image_ids.append(image_id)
        return image_ids
    
    
    def load_data(
            self, 
            image_id, 
            img_mode,
            shortest_side=84, 
            longest_side=84, 
            upscale_factor=1.0,
            resize_mode='pad_square',
            verbose=0
            ):
        """
        Loads preprocessed image, class ids and other information.

        Parameters
        ----------
        image_id : integer
            The image id from image_ids.
        img_mode : string
            Process image to either in 'rgb' or 'gray'.
        shortest_side, longest_side, upscale_factor, resize_mode : 
            See resize_image().
        verbose : binary, optional
            Whether to print out the preprocessing time. The default is 0.

        Returns
        -------
        image1 : numpy array, [resized_h, resized_w, 3] for RGB or 
        [resized_h, resized_w] for gray
            The preprocessed image.
        class_id : integer
            The class id for the image.
        cache : list
            Includes image path, class name, window, scale, padding, crop, 
            original image shape and preprocessed image shape.

        """
        # load image
        image_path = self.get_img_info(image_id)['path']
        t1 = time.time()
        image = load_image(image_path, img_mode)
        t2 = time.time()
        if verbose: 
            print('loading image: %fs' %(t2-t1))
        image_shape = image.shape
        
        # resize image
        t1 = time.time()
        image1, window, scale, padding, crop = resize_image(
            image, 
            shortest_side, 
            longest_side, 
            upscale_factor, 
            resize_mode)
        t2 = time.time()
        if verbose: 
            print('resize mode:', resize_mode)
            print('resizing image: %fs' %(t2-t1))

        # augment if needed
        class_name = self.get_img_info(image_id)['class_name']
        if 'rot' in class_name.split('_'):
            angle = int(class_name.split('_')[-1])
            k = angle / 90
            t1 = time.time()
            # image1 = skimage.transform.rotate(image1, angle)
            image1 = np.rot90(image1, k)
            t2 = time.time()
            if verbose: 
                print('rotating image: %fs' %(t2-t1))
        elif 'flip' in class_name.split('_'):
            if 'lr' in class_name.split('_'):
                t1 = time.time()
                image1 = np.fliplr(image1)
                t2 = time.time()
            if 'ud' in class_name.split('_'):
                t1 = time.time()
                image1 = np.flipud(image1)
                t2 = time.time()
            if verbose: 
                print('fliping image: %fs' %(t2-t1))
        
        cache = (
            image_path,
            class_name,
            window, 
            scale, 
            padding, 
            crop, 
            image_shape, 
            image1.shape) 
        class_id = self.class_name2id(class_name)
        return image1, class_id, cache
    
    
    