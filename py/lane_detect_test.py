#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 22:05:28 2019

@author: mingrui
"""
import os.path
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from moviepy.editor import VideoFileClip

from lane_detect_param import LaneDetectParam
from lane_detector import LaneDetector

# test lane detection on the images in the given path
def test_lane_detect(input_path):
    # load parameters
    param = LaneDetectParam()
    param.debug = True
    
    ld = LaneDetector(param)
    
    # input and result paths
    input_path = os.path.join(param.root_path, input_path)
    result_path = os.path.join(input_path, 'lane_detect')
    if not os.path.exists(result_path):
        os.mkdir(result_path)
        
    images = glob.glob(os.path.join(input_path, '*.jpg'))
    images.sort()
    for fname in images:        
        img = mpimg.imread(fname)    
        file_name = fname.split('/')[-1].split('.')[0].strip() + '.jpg'
            
        # lane detection and intermediate results
        print(file_name)
        if param.debug:
            undist_img, binary_img, warped_img, poly_fit_img, lane_img = ld.lane_detect(img)
        else:
            lane_img = ld.lane_detect(img)
            undist_img = None
            binary_img = None
            warped_img = None
            poly_fit_img = None
            
        # save results
        if param.debug:
            undist_fname = os.path.join(result_path, 'undist_' + file_name)
            mpimg.imsave(undist_fname, undist_img)
            
            bin_fname = os.path.join(result_path, 'bin_' + file_name)
            cv2.imwrite(bin_fname, binary_img*255)
            
            warp_fname = os.path.join(result_path, 'warp_' + file_name)
            cv2.imwrite(warp_fname, warped_img*255)
            
            if poly_fit_img is not None:
                fit_fname = os.path.join(result_path, 'fit_' + file_name)
                mpimg.imsave(fit_fname, poly_fit_img)
        
        lane_fname = os.path.join(result_path, 'lane_' + file_name)
        mpimg.imsave(lane_fname, lane_img)


def process_video(input_video, output_video):
    param = LaneDetectParam()
    param.debug = False
    ld = LaneDetector(param)
    
    clip = VideoFileClip(input_video)
    #clip = VideoFileClip(input_video).subclip(32, 45)
    white_clip = clip.fl_image(ld.lane_detect)
    white_clip.write_videofile(output_video, audio=False)
    
def test_video(input_video):
    path = input_video.split('/')
    path[-1] = 'ld_' + path[-1]
    output_video = '/'.join(path)
    process_video(input_video, output_video)
    
def save_frames(input_video, T1, T2):
    N = (T2-T1)*20
    clip1 = VideoFileClip(input_video).subclip(T1, T2)
    for t in range(N):
        clip1.save_frame('../frames/frame_' + str(1000+t) + '.jpg', t=(T2-T1)*t/N)


if __name__ == '__main__':
    path = 'frames'
    test_lane_detect(path)
    #test_video('../harder_challenge_video.mp4')
    #save_frames('../harder_challenge_video.mp4', 0, 4)