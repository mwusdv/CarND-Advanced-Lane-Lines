#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 21:00:37 2019

@author: mingrui
"""
import numpy as np

def compute_curvature(poly2_fit, y):
    a, b, c = poly2_fit
    curvature = (1+(2*a*y+b)**2)**1.5/2/np.abs(a)
    return curvature

def measure_curvature_pixels(ploty, left_fit, right_fit):
    '''
    Calculates the curvature of polynomial functions in pixels.
    '''
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    # calculation of R_curve (radius of curvature)
    left_curverad= compute_curvature(left_fit, y_eval)
    right_curverad = compute_curvature(right_fit, y_eval)
    
    return left_curverad, right_curverad

def measure_curvature_real(ploty, left_fit, right_fit, param):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
  
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    # calculation of R_curve (radius of curvature)
    left_curverad = compute_curvature(left_fit, y_eval*param.ym_per_pix)
    right_curverad = compute_curvature(right_fit, y_eval*param.ym_per_pix)
     
    return left_curverad, right_curverad