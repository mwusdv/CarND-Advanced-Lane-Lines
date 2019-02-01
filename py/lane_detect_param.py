#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 23:13:11 2019

@author: mingrui
"""

import cv2
import os.path
import pickle
import numpy as np

class LaneDetectParam:
    def __init__(self, debug=False, chessbd_corners_nx=9, chessbd_corners_ny=6, 
                 img_row=720, img_col=1280,
                 sobel_kernel_size=9, sobel_gradx_thresh=(20, 200), sobel_grady_thresh=(0, 255),
                 sobel_mag_thresh=(20, 100), sobel_dir_thresh=(0.7, 1.3),
                 s_channel_thresh=(50, 255), s_channel_lb = 0, 
                 white_rgb_threshold = np.array([0, 0, 0]), 
                 yellow_hsv_lb = np.array([15, 100, 180]),
                 yellow_hsv_ub = np.array([40, 255, 255]),
                 conv_window_width=50, conv_window_height=80, conv_margin=100, 
                 hist_nwindows=9, hist_margin=100, hist_minpix=50,
                 ym_per_pix=30/720, xm_per_pix=3.7/700):
        
        self.debug = debug
        
        # directories
        self.root_path = '/home/mingrui/Mingrui/udacity/self-driving/CarND-Advanced-Lane-Lines/'
        self.param_path = os.path.join(self.root_path, 'lane_decect_param.pkl')
        self.camera_cal_path = os.path.join(self.root_path, 'camera_cal/')
        
        # image size
        self.img_row = img_row
        self.img_col = img_col
        
        # number of chessboard corners for camera calibration
        self.chessbd_corners_nx = chessbd_corners_nx
        self.chessbd_corners_ny = chessbd_corners_ny
        self.calib_mtx = None
        self.calib_dist = None
        self.warp_mtx = None
        self.inv_warp_mtx = None
        
        # binarization based on gradient
        self.sobel_kernel_size = sobel_kernel_size
        self.sobel_gradx_thresh = sobel_gradx_thresh
        self.sobel_grady_thresh = sobel_grady_thresh
        self.sobel_mag_thresh = sobel_mag_thresh
        self.sobel_dir_thresh = sobel_dir_thresh
        
        # binary thresholding based on s-channel
        self.s_channel_thresh = s_channel_thresh
        self.s_channel_lb = s_channel_lb
        
        # for color detection
        self.yellow_hsv_lb = yellow_hsv_lb
        self.yellow_hsv_ub = yellow_hsv_ub
        
        self.white_rgb_threshold = white_rgb_threshold
        
        # for convolution
        self.conv_window_width = conv_window_width
        self.conv_window_height = conv_window_height
        self.conv_margin = conv_margin
        
        # lane fit based on histogram
        self.hist_nwindows = hist_nwindows
        self.hist_margin = hist_margin
        self.hist_minpix = hist_minpix
        
        # meters per pixel in y and x dimension
        self.ym_per_pix = ym_per_pix 
        self.xm_per_pix = xm_per_pix
        
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
        
        # record poly fit for lanes
        self.fit_gap_ub = 1e-3
        self.fit_momentum = 0.1 # to smooth the poly fit parameters
        self.record_fit = False  # true is for dealing with a sequence of frames
        self.left_fit = None
        self.right_fit = None
        self.left_fit_real = None
        self.right_fit_real = None
        
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
        
        
