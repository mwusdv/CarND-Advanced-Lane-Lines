#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 23:13:11 2019

@author: mingrui
"""

import cv2
import os.path
import pickle

class LaneDetectParam:
    def __init__(self):
        self.debug = False
        
        # directories
        self.root_path = '/home/mingrui/Mingrui/udacity/self-driving/CarND-Advanced-Lane-Lines/'
        self.param_path = os.path.join(self.root_path, 'lane_decect_param.pkl')
        self.camera_cal_path = os.path.join(self.root_path, 'camera_cal/')
        
        # image size
        self.img_row = 720
        self.img_col = 1280
        
        # number of chessboard corners for camera calibration
        self.chessbd_corners_nx = 9
        self.chessbd_corners_ny = 6
        self.calib_mtx = None
        self.calib_dist = None
        self.warp_mtx = None
        self.inv_warp_mtx = None
        
        # binarization based on gradient
        self.sobel_kernel_size = 9
        self.sobel_gradx_thresh = (20, 100)
        self.sobel_grady_thresh = (0, 255)
        self.sobel_mag_thresh = (20, 100)
        self.sobel_dir_thresh = (0.7, 1.3)
        
        # binary thresholding based on s-channel
        self.s_channel_thresh = (180, 255)
        self.s_channel_lb = 15
        
        # for convolution
        self.conv_window_width = 50
        self.conv_window_height = 80
        self.conv_margin = 100
        
        # lane fit based on histogram
        self.hist_nwindows = 9
        self.hist_margin = 100
        self.hist_minpix = 50
        
        # meters per pixel in y and x dimension
        self.ym_per_pix = 30/720 
        self.xm_per_pix = 3.7/700
        
        # poly line drawing
        self.poly_line_color = (255, 255, 0)
        self.poly_line_thickness = 2
        
        # draw lane and lane region
        self.left_lane_color = (255, 0, 0)
        self.right_lane_color = (0, 0, 255)
        self.draw_lane_thickness = 20
        self.lane_region_color = (0, 255, 0)
    
        # text on the lane image
        self.first_line_pos = (300, 25)
        self.line_gap = 30
        self.text_color = (255, 255, 255)
        self.font_face = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1
        
        self.load_calib_param()
        
    def load_calib_param(self):     
        if not os.path.exists(self.param_path):
            return
        
        fd = open(self.param_path, 'rb')
        self.calib_mtx = pickle.load(fd)
        self.calib_dist = pickle.load(fd)
        self.warp_mtx = pickle.load(fd)
        self.inv_warp_mtx = pickle.load(fd)
        fd.close()
        
    def save_calib_param(self, path=''):
        if len(path) > 0:
            self.param_path = path
            
        fd = open(self.param_path, 'wb')
        pickle.dump(self.calib_mtx, fd)
        pickle.dump(self.calib_dist, fd)
        pickle.dump(self.warp_mtx, fd)
        pickle.dump(self.inv_warp_mtx, fd)
        fd.close()
        
        
