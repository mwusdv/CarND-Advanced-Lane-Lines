#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 23:13:11 2019

@author: mingrui
"""

import os.path
import pickle

class LaneDetectParam:
    def __init__(self, debug=False, chessbd_corners_nx=9, chessbd_corners_ny=6, 
                 sobel_kernel_size=9, sobel_gradx_thresh=(20, 100), sobel_grady_thresh=(0, 255),
                 sobel_mag_thresh=(20, 100), sobel_dir_thresh=(0.7, 1.3),
                 s_channel_thresh=(180, 255)):
        
        self.debug = debug
        
        # directories
        self.root_path = '/home/mingrui/Mingrui/udacity/self-driving/CarND-Advanced-Lane-Lines/'
        self.param_path = os.path.join(self.root_path, 'lane_decect_param.pkl')
        self.camera_cal_path = os.path.join(self.root_path, 'camera_cal/')
        
        # image size
        self.img_row = 0
        self.img_col = 0
        
        # number of chessboard corners for camera calibration
        self.chessbd_corners_nx = chessbd_corners_nx
        self.chessbd_corners_ny = chessbd_corners_ny
        self.calib_mtx = None
        self.calib_dist = None
        
        # binarization based on gradient
        self.sobel_kernel_size = sobel_kernel_size
        self.sobel_gradx_thresh = sobel_gradx_thresh
        self.sobel_grady_thresh = sobel_grady_thresh
        self.sobel_mag_thresh = sobel_mag_thresh
        self.sobel_dir_thresh = sobel_dir_thresh
        
        # binary thresholding based on s-channel
        self.s_channel_thresh = s_channel_thresh
        
        
    def load(self, path=''):
        if len(path) > 0:
            self.param_path = path
            
        fd = open(self.param_path, 'rb')
        param = pickle.load(fd)
        fd.close()
        return param
        
        
    def save(self, path=''):
        if len(path) > 0:
            self.param_path = path
            
        fd = open(self.param_path, 'wb')
        pickle.dump(self, fd)
        fd.close()
        
def load_param(path=''):
    param = LaneDetectParam()
    return param.load(path)
    