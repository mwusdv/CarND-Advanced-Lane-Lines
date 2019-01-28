#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 21:56:08 2019

@author: mingrui
"""

import os.path
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
from lane_detect_param import LaneDetectParam


def compute_undistort_param(param):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((param.chessbd_corners_nx * param.chessbd_corners_ny, 3), np.float32)
    objp[:,:2] = np.mgrid[0:param.chessbd_corners_nx, 0:param.chessbd_corners_ny].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    
    # Make a list of chessbd_cornersration images
    images = glob.glob(os.path.join(param.camera_cal_path, '*.jpg'))
    
    # Step through the list and search for chessboard corners
    img_size = None
    for idx, fname in enumerate(images):
        #print(fname)
        img = cv2.imread(fname)
        if idx == 0:
            param.img_row = img.shape[0]
            param.img_col = img.shape[1]
            img_size = (param.img_col, param.img_row)
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (param.chessbd_corners_nx, param.chessbd_corners_ny), None)
    
        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
    
            # Draw and display the corners
            if param.debug:
                cv2.drawChessboardCorners(img, (param.chessbd_corners_nx, param.chessbd_corners_ny), corners, ret)
                plt.imshow(img)
                plt.show()
                    

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    param.calib_mtx = mtx
    param.calib_dist = dist       

    return param

def compute_perpective_transform_param(param):
    #src = np.float32([[246, 694], [1062, 694], [686, 454], [598, 454]])
    src = np.float32([[246, 694], [1062, 694], [752, 493], [537, 493]])
    dst = np.array([[src[0][0], param.img_row-1], [src[1][0], param.img_row-1], [src[1][0], 0], [src[0][0], 0]], dtype=np.float32)
    
    #transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
        
    param.warp_mtx = M
    
    return param

               
def camera_calibration():
    param = LaneDetectParam()
    
    # parameters for distoration correction
    param = compute_undistort_param(param)
    
    # parameters for perspective transformation
    param = compute_perpective_transform_param(param)
    
    param.save_calib_param()
    
    return param


def test_camera_calibration(param, input_path, do_warp):
    # result directory
    input_path = os.path.join(param.root_path, input_path)
    result_path = os.path.join(input_path, 'calib_result')
    if not os.path.exists(result_path):
        os.mkdir(result_path)
      
    # output calibration output
    images = glob.glob(os.path.join(input_path, '*.jpg'))
    for fname in images:
        img = cv2.imread(fname)    
        file_name = fname.split('/')[-1].split('.')[0].strip()
        
        # distortion correction
        undist = cv2.undistort(img, param.calib_mtx, param.calib_dist, None, param.calib_mtx)
        undist_fname = os.path.join(result_path, 'undist_' + file_name + '.jpg')
        cv2.imwrite(undist_fname, undist)
       
        # perspective transform
        if do_warp:
            warped = cv2.warpPerspective(undist, param.warp_mtx, (param.img_col, param.img_row))
            warped_fname = os.path.join(result_path, 'warped_' + file_name + '.jpg')    
            cv2.imwrite(warped_fname, warped)
            
        
if __name__ == '__main__':
    param = camera_calibration()
    test_camera_calibration(param, 'camera_cal', do_warp=False)
    test_camera_calibration(param, 'test_images', do_warp=True)