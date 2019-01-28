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

from lane_detect_param import LaneDetectParam

#from  camera_calibration import *
from binary_thresh import binary_thresholding

from lane_find_conv import lane_find_conv
from lane_find_hist import fit_polynomial
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
    warped_img = cv2.warpPerspective(binary_img, param.warp_mtx, (param.img_col, param.img_row))
    
    # detect lane pixels and fit to find the lane
    poly_fit_img, ploty, left_fitx, right_fitx = fit_polynomial(warped_img, param)

    
    #lane_find_conv(warped_img, param)
    #lane_img = lane_fit(birds_eye_img, param)
    
    # compute curvature and vheicle position with repsecto to center
    #curvature, position = cure_pos(lane_img, param)
    
    # wapr the detected lane boundaries back onto the original image
    #lane_img_warp_back = warp_back(lane_img, curvature, position, param)
    
    return undist_img, binary_img, warped_img, poly_fit_img

# test lane detection on the images in the given path
def test_lane_detect(input_path):
    # load parameters
    param = LaneDetectParam()
    param.debug = True
    
    # input and result paths
    input_path = os.path.join(param.root_path, input_path)
    result_path = os.path.join(input_path, 'lane_detect')
    if not os.path.exists(result_path):
        os.mkdir(result_path)
        
    images = glob.glob(os.path.join(input_path, '*.jpg'))
    for fname in images:
        img = mpimg.imread(fname)    
        file_name = fname.split('/')[-1].split('.')[0].strip() + '.jpg'
        
        # lane detection and intermediate results
        print(file_name)
        undist_img, binary_img, warped_img, poly_fit_img = lane_detect(img, param)
        
        # save results
        undist_fname = os.path.join(result_path, 'undist_' + file_name)
        mpimg.imsave(undist_fname, undist_img)
        
        bin_fname = os.path.join(result_path, 'bin_' + file_name)
        cv2.imwrite(bin_fname, binary_img*255)
        
        warp_fname = os.path.join(result_path, 'warp_' + file_name)
        cv2.imwrite(warp_fname, warped_img*255)
        
        fit_fname = os.path.join(result_path, 'fit_' + file_name)
        mpimg.imsave(fit_fname, poly_fit_img)

if __name__ == '__main__':
    path = 'test_images'
    test_lane_detect(path)