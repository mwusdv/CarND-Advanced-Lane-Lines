#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 20:30:20 2019

@author: mingrui
"""
import os.path
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from lane_detect_param import load_param

#from  camera_calibration import *
from binary_thresh import binary_thresholding
#from lane_find_hist import *

#def lane_detect(img, param):
    # distortion correction
#    undist = cv2.undistort(img, param.mtx, param.dist, None, param.mtx)
    
    # binary thresholding based on HLS and gradient
#    binary_img = hls_gradient_select(undist, param)
    
    # warp

def get_interest_region_vertices(img, param):
    row, col = img.shape[:2]
    mid = col//2
 
    # height of the interest region
    height = int(row * param.ir_row_ratio)
    
    # length of the upper and lower horizontal edges
    upper_edge_length = int(col * param.ir_upper_col_ratio)
    lower_edge_length = int(col * param.ir_lower_col_ratio)
    offset = (col - lower_edge_length) // 2
    
    # trapezoid, (row, col)
    vertices = np.array([[offset, row-1], [col-1-offset, row-1], \
                          [mid+upper_edge_length/2, row-height], [mid-upper_edge_length/2, row-height]], dtype=np.float32)
    
    
    return vertices


# lane detection pipeline
def lane_detect(img, param):
    # param already contains the pre-computed camersa 
    # calibration matrix and distortion coefficitns. 
    # In addition, param also contains the source and 
    # destination points for the perspective transformation
    
    # distortion correction
    undist_img = cv2.undistort(img, param.calib_mtx, param.calib_dist, None, param.calib_mtx)
    
    # binary thresholding based on gradient and color
    binary_img = binary_thresholding(undist_img, param)
    
    #perspective transformation to get eh birde-eye view
    birds_eye_img = cv2.warpPerspective(binary_img, param.warp_mtx, (param.img_col, param.img_col))
    
    # detect lane pixels and fit to find the lane
    #lane_img = lane_fit(birds_eye_img, param)
    
    # compute curvature and vheicle position with repsecto to center
    #curvature, position = cure_pos(lane_img, param)
    
    # wapr the detected lane boundaries back onto the original image
    #lane_img_warp_back = warp_back(lane_img, curvature, position, param)
    
    return undist_img, binary_img, birds_eye_img

# test lane detection on the images in the given path
def test_lane_detect(input_path):
    # load parameters
    param = load_param()
    
    # input and result paths
    input_path = os.path.join(param.root_path, input_path)
    result_path = os.path.join(input_path, 'lane_detect')
    if not os.path.exists(result_path):
        os.mkdir(result_path)
        
    images = glob.glob(os.path.join(input_path, '*.jpg'))
    for fname in images:
        img = mpimg.imread(fname)    
        file_name = fname.split('/')[-1].split('.')[0].strip()
        
        # lane detection and intermediate results
        undist_img, binary_img, birds_eye_img = lane_detect(img, param)
        
        # save results
        undist_fname = os.path.join(result_path, 'undist_' + file_name + '.jpg')
        mpimg.imsave(undist_fname, undist_img)
        
        bin_fname = os.path.join(result_path, 'bin_' + file_name + '.jpg')
        cv2.imwrite(bin_fname, binary_img*255)
        
        birds_fname = os.path.join(result_path, 'birds_' + file_name + '.jpg')
        cv2.imwrite(birds_fname, birds_eye_img*255)
        
        
def lane_detect0(image, path, result_path, param):
    img = mpimg.imread(fname)
    file_name = fname.split('/')[1]
        
    undist = cv2.undistort(img, param.mtx, param.dist, None, param.mtx)
    undist_path = os.path.join(result_path, 'undist_' + file_name)
    mpimg.imsave(undist_path, undist)

    # binary thresholding based on HLS and gradient
    binary_img = hls_gradient_select(undist, param)
    binary_path = os.path.join(result_path, 'binary_' + file_name)
    cv2.imwrite(binary_path, binary_img*255)
    
    bin_warped = warp(binary_img, param)
    warp_path = os.path.join(result_path, 'warp_' + file_name)
    cv2.imwrite(warp_path, bin_warped*255)
    
    lane_fit_img = fit_polynomial(bin_warped)
    lf_path = os.path.join(result_path, 'lf_' + file_name)
    mpimg.imsave(lf_path, lane_fit_img)
    
    
def test_pipeline(path, param):
    image_path = os.path.join(path, '*.jpg')
    images = glob.glob(image_path)
    result_path = os.path.join(path, 'warp')
    for fname in images:
        print(fname)
        lane_detect(fname, path, result_path, param)
      
        

if __name__ == '__main__':
    path = 'test_images'
    test_lane_detect(path)