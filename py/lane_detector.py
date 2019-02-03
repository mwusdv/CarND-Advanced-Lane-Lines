#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 17:37:02 2019

@author: mingrui
"""

import cv2

from lane_detect_param import LaneDetectParam
from binary_thresholder import BinaryThresholder

class LaneDetector:
    def __init__(self, param):
        self._param = param
        self._bsh = BinaryThresholder(param)
        
        
    # lane detection pipeline
    def lane_detect(self, img):
        # param already contains the pre-computed camersa 
        # calibration matrix and distortion coefficitns. 
        # In addition, param also contains the source and 
        # destination points for the perspective transformation
        
        # distortion correction
        undist_img = cv2.undistort(img, self.param.calib_mtx, self.param.calib_dist, None, self.param.calib_mtx)
        
        # binary thresholding based on gradient and color
        binary_img = self._bsh.binary_thresholding(undist_img)
        
        #perspective transformation to get eh birde-eye view
        warped_img = cv2.warpPerspective(binary_img, self._param.warp_mtx, (self._param.img_col, self._param.img_row))
        
        # detect lane pixels and fit to find the lane
        left_fit, right_fit, left_fit_real, right_fit_real, ploty, poly_fit_img = fit_polynomial(warped_img, param)
    
        # vehcile offset with respect to the center
        offset = compute_vehicle_pos(ploty, left_fit, right_fit, param)
        
        left_curverad, right_curverad = measure_curvature_real(ploty, left_fit_real, right_fit_real, param)
    
        curverad = (left_curverad + right_curverad)/2
        lane_img = render_lane(img, ploty, left_fit, right_fit, curverad, offset, param)
        #lane_find_conv(warped_img, param)
        
        # compute curvature and vheicle position with repsecto to center
        #curvature, position = cure_pos(lane_img, param)
        
        # wapr the detected lane boundaries back onto the original image
        #lane_img_warp_back = warp_back(lane_img, curvature, position, param)
        
        return undist_img, binary_img, warped_img, poly_fit_img, lane_img, left_curverad, right_curverad, offset