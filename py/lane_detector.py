#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 17:37:02 2019

@author: mingrui
"""

import cv2
import numpy as np

from binary_thresholder import BinaryThresholder
from poly_fitter import PolyFitter

class LaneDetector:
    def __init__(self, param):
        self._param = param
        self._bsh = BinaryThresholder(param)
        self._poly_fitter = PolyFitter(param)
        
        
    def compute_vehicle_pos(self):
        y_eval = np.max(self._poly_fitter._ploty)
        left_x = self._poly_fitter._left_fit[0]*y_eval**2 + self._poly_fitter._left_fit[1]*y_eval + self._poly_fitter._left_fit[2]
        right_x = self._poly_fitter._right_fit[0]*y_eval**2 + self._poly_fitter._right_fit[1]*y_eval + self._poly_fitter._right_fit[2]
        
        offset = (left_x + right_x)/2 - self._param.img_col/2
        offset *= self._poly_fitter._param.xm_per_pix
        
        return offset

    def render_lane(self, color_img, curverad, offset):
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(color_img, dtype=np.uint8)
    
        # left and right lanes
        left_pts = self._poly_fitter._left_pts
        N = self._poly_fitter._right_pts.shape[0]
        right_pts = self._poly_fitter._right_pts[range(N-1,-1,-1), :] # reverse for plotting
        

        # draw left and right lanes
        cv2.polylines(warp_zero, [left_pts], False, self._param.left_lane_color, self._param.draw_lane_thickness)
        cv2.polylines(warp_zero, [right_pts], False, self._param.right_lane_color, self._param.draw_lane_thickness)
        
        # fill thelane rgion
        cv2.fillPoly(warp_zero, [np.vstack((left_pts, right_pts))], self._param.lane_region_color)
        
        # warp back to the original image
        back_warped = cv2.warpPerspective(warp_zero, self._param.inv_warp_mtx, (self._param.img_col, self._param.img_row))
        lane_img = cv2.addWeighted(color_img, 0.8, back_warped, 1.0, 0)
        
        # show curvature
        text = 'Radius of Curvature = ' + '{:.2f}'.format(curverad) + ' (m)'
        text_pos= self._param.first_line_pos
        cv2.putText(lane_img, text, text_pos, self._param.font_face, self._param.font_scale, self._param.text_color)
        
        offset_str = 'left' if offset < 0 else 'right'
        text = 'Vheicle is ' + '{:.2f}'.format(abs(offset)) + ' m ' + offset_str + ' of center'
        text_pos = (text_pos[0], text_pos[1] + self._param.line_gap)
        cv2.putText(lane_img, text, text_pos, self._param.font_face, self._param.font_scale, self._param.text_color)
        
        return lane_img 

    # lane detection pipeline
    def lane_detect(self, img):
        # param already contains the pre-computed camersa 
        # calibration matrix and distortion coefficitns. 
        # In addition, param also contains the source and 
        # destination points for the perspective transformation
        
        # distortion correction
        undist_img = cv2.undistort(img, self._param.calib_mtx, self._param.calib_dist, None, self._param.calib_mtx)
        
        # binary thresholding based on gradient and color
        binary_img = self._bsh.binary_thresholding(undist_img)
        
        #perspective transformation to get eh birde-eye view
        warped_img = cv2.warpPerspective(binary_img, self._param.warp_mtx, (self._param.img_col, self._param.img_row))
        
        # detect lane pixels and fit to find the lane
        poly_fit_img = self._poly_fitter.fit_polynomial(warped_img)
        
        # vehcile offset with respect to the center
        offset = self.compute_vehicle_pos()
      
        # curvature
        curverad = (self._poly_fitter._left_curverad + self._poly_fitter._right_curverad)/2
        
        # render lane image
        lane_img = self.render_lane(img, curverad, offset)
        
        if self._param.debug:
            return undist_img, binary_img, warped_img, poly_fit_img, lane_img
        else:
            return lane_img