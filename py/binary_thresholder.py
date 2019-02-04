#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 23:16:14 2019

@author: mingrui
"""

import os.path
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
from lane_detect_param import LaneDetectParam

class BinaryThresholder:
    def __init__(self, param):
        self._param = param
        
    # directional gradients
    def compute_sobel(self, gray):
         # Calculate gradient with respect to x and y respectively
         self._sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self._param.sobel_kernel_size)
         self._sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self._param.sobel_kernel_size)
         
         self._abs_sx = np.abs(self._sx)
         self._abs_sy = np.abs(self._sy)
         
         self._mag = np.sqrt(np.square(self._sx) + np.square(self._sy))
         
    # color space
    def s_space(self, img):
        # HLS color space
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        self._s_channel = hls[:,:,2]
         
    # thresholding based on directional gradient
    def abs_sobel_thresh(self, orient):
        # Calculate directional gradient
        if orient == 'x':
            abs_sobel = self._abs_sx
            grad_thresh = self._param.sobel_gradx_thresh
        else:
            abs_sobel = self._abs_sy
            grad_thresh = self._param.sobel_grady_thresh
        
        # normalization
        sobel_img = np.uint8(abs_sobel*255 / np.max(abs_sobel))
  
        # thresholding      
        grad_binary = np.zeros_like(sobel_img, dtype=np.uint8)
        grad_binary[(sobel_img >= grad_thresh[0]) & (sobel_img <= grad_thresh[1])] = 1
        
        return grad_binary

    # thresholding based on magnituded of the gradient
    def mag_thresh(self):
        # normalization
        mag_img = np.uint8(self._mag*255/np.max(self._mag))
    
        # thresholding
        mag_binary = np.zeros_like(mag_img, dtype=np.uint8)
        mag_binary[(mag_img >= self._param.sobel_mag_thresh[0]) & (mag_img <= self._param.sobel_mag_thresh[1])] = 1
        
        return mag_binary

    # thresholding based on gradient direction
    def dir_threshold(self):
        # Calculate gradient direction
        direction = np.arctan2(self._abs_sy, self._abs_sx)
        
        # thresholding
        dir_binary = np.zeros_like(direction, dtype=np.uint8)
        dir_binary[(direction >= self._param.sobel_dir_thresh[0]) & (direction <= self._param.sobel_dir_thresh[1])] = 1
        
        return dir_binary
        
    # thresholding based on sobel gradient
    def sobel_binarization(self):
        # based on directional gradient
        gradx = self.abs_sobel_thresh('x')
        grady = self.abs_sobel_thresh('y')
        
        # based on magnitude
        mag_binary = self.mag_thresh()
        
        # based on gradient direction
        dir_binary = self.dir_threshold()
    
        # combine all the results
        sobel_binary = np.zeros_like(dir_binary)
        sobel_binary[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
        
        return sobel_binary
             
    # thresholding based on s-channel in hls space
    def hls_s_select(self):
        # Apply a threshold to the S channel
        s_binary = np.zeros_like(self._s_channel)
        s_binary[(self._s_channel > self._param.s_channel_thresh[0]) \
                 & (self._s_channel <= self._param.s_channel_thresh[1])] = 1
        

        return s_binary

    # binary thresholding based on both color and gradient
    def binary_thresholding(self, img):
        # preparations 
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        self.compute_sobel(gray)
        self.s_space(img)
        
        # thresholding based on gradient and color
        sobel_binary = self.sobel_binarization()
        
        # binaray thresholding based on HLS
        s_binary = self.hls_s_select()
        
        # combination
        binary_output = np.zeros_like(sobel_binary)
        binary_output[((sobel_binary == 1) | (s_binary == 1)) \
                      & (self._s_channel > self._param.s_channel_lb)] = 1
                       
        N = binary_output.shape[0] * binary_output.shape[1]
        if np.sum(binary_output) / N > 0.4:
            binary_output[(np.max(img, axis=2) <= self._param.rgb_max_lb)] = 0
            
        return binary_output

def test_binary_thresholding(input_path, param):
    # result directory
    input_path = os.path.join(param.root_path, input_path)
    result_path = os.path.join(input_path, 'bin_result')
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    
    # test images
    images = glob.glob(os.path.join(input_path, '*.jpg'))
    
    bsh = BinaryThresholder(param)
    for fname in images:
        # binary thresholding
        img = mpimg.imread(fname)
        binary_img = bsh.binary_thresholding(img)
        
        # save result
        file_name = fname.split('/')[-1].split('.')[0].strip()
        bin_fname = os.path.join(result_path, 'bin_' + file_name + '.jpg')
        cv2.imwrite(bin_fname, binary_img*255)

            
if __name__ == '__main__':
    param = LaneDetectParam()
    input_path = 'test_images'
    test_binary_thresholding(input_path, param)

