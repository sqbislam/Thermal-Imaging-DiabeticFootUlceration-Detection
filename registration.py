# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 19:37:51 2022

@author: Saqib
"""

# third party imports
import tensorflow as tf
assert tf.__version__.startswith('2.')
tf.compat.v1.enable_eager_execution()

# local imports
import voxelmorph as vxm



class RegistrationModel:
    def __init__(self):
        tf.compat.v1.enable_eager_execution()

        self.model = self.get_vxm_model()
    def get_vxm_model(self):
        # configure unet input shape (concatenation of moving and fixed images)
        ndim = 2
        unet_input_features = 2
        inshape = (*(128, 128), unet_input_features)
        
        # configure unet features 
        nb_features = [
            [32, 32, 32, 32, 32],         # encoder features
            [32, 32, 32, 32, 16]  # decoder features
        ]
        
        # build model
        unet = vxm.networks.Unet(inshape=inshape, nb_features=nb_features)
        
        # transform the results into a flow field.
        disp_tensor = tf.keras.layers.Conv2D(ndim, kernel_size=3, padding='same', name='disp')(unet.output)
        
        # check tensor shape
        print('displacement tensor:', disp_tensor.shape)
        

        # build transformer layer
        spatial_transformer = vxm.layers.SpatialTransformer(name='transformer')
        
        # extract the first frame (i.e. the "moving" image) from unet input tensor
        moving_image = tf.expand_dims(unet.input[..., 0], axis=-1)
        
        # warp the moving image with the transformer
        moved_image_tensor = spatial_transformer([moving_image, disp_tensor])
    
        outputs = [moved_image_tensor, disp_tensor]
        vxm_model = tf.keras.models.Model(inputs=unet.inputs, outputs=outputs)
        
        # build model using VxmDense
        inshape = (128, 128)
        vxm_model = vxm.networks.VxmDense(inshape, nb_features, int_steps=0)
        
        
        # voxelmorph has a variety of custom loss classes
        losses = [vxm.losses.NCC().loss, vxm.losses.Grad('l1').loss]
        
        # usually, we have to balance the two losses by a hyper-parameter
        lambda_param = 0.17
        loss_weights = [1, lambda_param]
        
        vxm_model.compile(optimizer='Adam', loss=losses, loss_weights=loss_weights)
        
        print('input shape: ', ', '.join([str(t.shape) for t in vxm_model.inputs]))
        print('output shape:', ', '.join([str(t.shape) for t in vxm_model.outputs]))
        return vxm_model