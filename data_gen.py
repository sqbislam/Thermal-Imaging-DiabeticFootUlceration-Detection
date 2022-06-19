# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 19:39:49 2022

@author: Saqib
"""

# imports
import cv2 as cv

# third party imports
import numpy as np
import tensorflow as tf
assert tf.__version__.startswith('2.')
tf.compat.v1.enable_eager_execution()


    

class DataGenerator:
    
    def vxm_data_generator(input_file, batch_size=1, log=False, index=0):
        """
        Takes input file name and renders input and output
        
        inputs:  moving [bs, H, W, 1], fixed image [bs, H, W, 1]
        outputs: moved image [bs, H, W, 1], zero-gradient [bs, H, W, 2]
        """
        
        fixed = cv.imread("./images/Left-GT.png",cv.IMREAD_GRAYSCALE) / 255
        test = cv.imread("./images/test/CG014_M_L.png",cv.IMREAD_GRAYSCALE) / 255
        
        # preliminary sizing
        vol_shape = fixed.shape # extract data shape
        ndims = len(vol_shape)
        
        
        # prepare a zero array the size of the deformation
        # we'll explain this below
        zero_phi = np.zeros([batch_size, *vol_shape, ndims])
        
        
        moving_images = test[np.newaxis, ..., np.newaxis]
        # idx2 = np.random.randint(0, gt_images.shape[0], size=batch_size)
        fixed_images = fixed[np.newaxis, ..., np.newaxis]
        inputs = [moving_images, fixed_images]
        
            
        # prepare outputs (the 'true' moved image):
        # of course, we don't have this, but we know we want to compare 
        # the resulting moved image with the fixed image. 
        # we also wish to penalize the deformation field. 
        
        outputs = [fixed_images, zero_phi]
        
        return (inputs,outputs)
    


