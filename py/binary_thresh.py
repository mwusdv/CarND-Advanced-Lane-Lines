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
from lane_detect_param import load_param

def abs_sobel_thresh(gray, orient, param):
    # Calculate directional gradient
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=param.sobel_kernel_size)
        grad_thresh = param.sobel_gradx_thresh
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=param.sobel_kernel_size)
        grad_thresh = param.sobel_grady_thresh
    
    abs_sobel = np.absolute(sobel)
    sobel_img = np.uint8(abs_sobel*255 / np.max(abs_sobel))
    
    grad_binary = np.zeros_like(sobel_img, dtype=np.uint8)
    grad_binary[(sobel_img >= grad_thresh[0]) & (sobel_img <= grad_thresh[1])] = 1
    
    return grad_binary

def mag_thresh(gray, param):
    # Calculate gradient magnitude
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=param.sobel_kernel_size)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=param.sobel_kernel_size)
    mag = np.sqrt(np.square(sx) + np.square(sy))
    
    mag_img = np.uint8(mag*255/np.max(mag))
    
    mag_binary = np.zeros_like(mag_img, dtype=np.uint8)
    mag_binary[(mag_img >= param.sobel_mag_thresh[0]) & (mag_img <= param.sobel_mag_thresh[1])] = 1
    
    return mag_binary

def dir_threshold(gray, param):
    # Calculate gradient direction
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=param.sobel_kernel_size)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=param.sobel_kernel_size)
    direction = np.arctan2(np.absolute(sy), np.absolute(sx))
    
    dir_binary = np.zeros_like(direction, dtype=np.uint8)
    dir_binary[(direction >= param.sobel_dir_thresh[0]) & (direction <= param.sobel_dir_thresh[1])] = 1
    
    return dir_binary


def sobel_binarization(gray, param):
    gradx = abs_sobel_thresh(gray, 'x', param)
    grady = abs_sobel_thresh(gray, 'y', param)
    mag_binary = mag_thresh(gray, param)
    dir_binary = dir_threshold(gray, param)

    binary_output = np.zeros_like(dir_binary)
    binary_output[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    
    return binary_output

def test_sobel_binarization(path, param):
    image_path = os.path.join(path, '*.jpg')
    images = glob.glob(image_path)
    for fname in images:
        print(fname)
        img = mpimg.imread(fname)
        binary_img = sobel_binarization(img, param)
        
        file_name = fname.split('/')[1]
        result_path = os.path.join(path, 'sobel_binary', file_name)
        cv2.imwrite(result_path, binary_img*255)
        
# select based on s-channel in hls space
def hls_s_select(img, param):
    # HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    
    # Apply a threshold to the S channel
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > param.s_channel_thresh[0]) & (s_channel <= param.s_channel_thresh[1])] = 1
    
    return binary_output

    
# binary thresholding based on both color and gradient
def binary_thresholding(img, param):
    # binary thresholding based on gradient
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    bin_sobel = sobel_binarization(gray, param)
    
    # binaray thresholding based on HLS
    bin_hls = hls_s_select(img, param)
    
    # combination
    binary_output = np.zeros_like(bin_sobel)
    binary_output[(bin_sobel == 1) | (bin_hls == 1)] = 1
    return binary_output

def test_binary_thresholding(input_path, param):
    # result directory
    input_path = os.path.join(param.root_path, input_path)
    result_path = os.path.join(input_path, 'bin_result')
    if not os.path.exists(result_path):
        os.mkdir(result_path)
      
    images = glob.glob(os.path.join(input_path, '*.jpg'))
    for fname in images:
        img = mpimg.imread(fname)
        file_name = fname.split('/')[-1].split('.')[0].strip()
        
        binary_img = binary_thresholding(img, param)
        
        bin_fname = os.path.join(result_path, 'bin_' + file_name + '.jpg')
        cv2.imwrite(bin_fname, binary_img*255)

        
if __name__ == '__main__':
    param = load_param()
    input_path = 'test_images'
    test_binary_thresholding(input_path, param)

