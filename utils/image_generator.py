#!/usr/bin/env/ python

# This Python script contains the ImageGenrator class.

#################################
# author = Drew Afromsky        #
# email = daa2162@columbia.edu  #
#################################

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.interpolation import rotate
from scipy.ndimage import gaussian_filter
from scipy import misc
import time
import cv2


class ImageGenerator:
    
    def show(self, images):

        """
        Plot the top 16 images (index 0~15) for visualization.
        :param images: images to be shown
        """

        xshow = images[:25]
        fig = plt.figure(figsize=(6,6))
        fig.set_tight_layout(True)

        for i in range(25):
            ax = fig.add_subplot(5,5,i+1)
            # ax.imshow((xshow[i,:]*255).astype(np.uint8))
            ax.imshow(xshow[i,:])

            ax.axis('off')

    def rotate(self, images, angle=0.0):
        
        """
        Rotate self.x by the angles (in degree) given.
        :param angle: Rotation angle in degrees.
        :return rotated: rotated dataset
        """
        
        self.dor = angle
        rotated = rotate(images, angle, reshape=False, axes=(1,2))
        print('Currrent rotation: {} degrees'.format(self.dor))
        return rotated
    
    def gauss_filt(self, images):
        res = []
        
        start_blur = time.time()
        for i in range(images.shape[0]):
            result = gaussian_filter(images[i], sigma=1)
            res.append(result)
        end_blur = time.time()
        serial_time_blur = end_blur - start_blur
        print(serial_time_blur)
        print(len(res))
        print(res[0].shape)

        fig = plt.figure(figsize=(6,6))
    
        for j in range(25):
            ax = fig.add_subplot(5,5,j+1)
            ax.imshow(res[j])
            ax.axis('off')

        return res
    
    def avg_blur(self, images):
        res_avg = []
        
        start_blur_avg = time.time()
        for i in range(images.shape[0]):
            result_avg = cv2.blur(images[i],(3,3))
            res_avg.append(result_avg)
        end_blur_avg = time.time()
        serial_time_blur_avg = end_blur_avg - start_blur_avg
        print(serial_time_blur_avg)
        print(len(res_avg))
        print(res_avg[0].shape)

        fig = plt.figure(figsize=(6,6))
        fig.set_tight_layout(True)
        
        for j in range(25):
            ax = fig.add_subplot(5,5,j+1)
            ax.imshow(res_avg[j])
            ax.axis('off')

        return res_avg

    def matrix_transpose(self, images):
        
        trans=[]

        start_trans = time.time()
        for i in range(images.shape[0]):
            img_T = np.transpose(images[i], (1,0,2))     
            trans.append(img_T) 
        end_trans = time.time()
        serial_time_trans = end_trans - start_trans
        print(serial_time_trans)
        print(len(trans))
        print(trans[0].shape)

        fig = plt.figure(figsize=(6,6))

        for j in range(25):
            ax = fig.add_subplot(5,5,j+1)
            ax.imshow(trans[j])
            ax.axis('off')

        return trans
