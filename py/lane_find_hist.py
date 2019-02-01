import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
from lane_detect_param import LaneDetectParam


def find_lane_pixels(binary_warped, param):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
     
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//param.hist_nwindows)
    
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(param.hist_nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        
        win_xleft_low = leftx_current - param.hist_margin  
        win_xleft_high = leftx_current+ param.hist_margin
        win_xright_low = rightx_current - param.hist_margin
        win_xright_high = rightx_current+ param.hist_margin
        
        # Draw the windows on the visualization image
        #if param.debug:
        #    cv2.rectangle(fit_img,(win_xleft_low,win_y_low),
        #    (win_xleft_high,win_y_high),(0,255,0), 2) 
            
       #     cv2.rectangle(fit_img,(win_xright_low,win_y_low),
       #     (win_xright_high,win_y_high),(0,255,0), 2) 
            
        # Identify the nonzero pixels in x and y within the window 
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                          (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # update left and right center
        if good_left_inds.shape[0] > param.hist_minpix:
            leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
        
        if good_right_inds.shape[0] > param.hist_minpix:
            rightx_current = np.int32(np.mean(nonzerox[good_right_inds])) 
            
    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty


def polyx(y, poly_fit):
    return poly_fit[0] * y**2 + poly_fit[1] * y + poly_fit[2]

def search_around_poly(binary_warped, param):
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    leftx = polyx(nonzeroy, param.left_fit)
    left_lane_inds = ((nonzerox > leftx-param.hist_margin) & (nonzerox < leftx+param.hist_margin))
    
    rightx = polyx(nonzeroy, param.right_fit)
    right_lane_inds = ((nonzerox > rightx-param.hist_margin) & (nonzerox < rightx+param.hist_margin))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
   
    return leftx, lefty, rightx, righty

    
def fit_polynomial(binary_warped, param):
    # Find our lane pixels first
    if param.left_fit is not None and param.right_fit is not None:
        leftx, lefty, rightx, righty = search_around_poly(binary_warped, param)
    else:
        leftx, lefty, rightx, righty = find_lane_pixels(binary_warped, param)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    
    #Fit a second order polynomial
    if len(leftx) == 0 or len(rightx) == 0:
        return None, None, None, None, ploty, None
    
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    left_fit_real = np.polyfit(lefty*param.ym_per_pix, leftx*param.xm_per_pix, 2)
    right_fit_real = np.polyfit(righty*param.ym_per_pix, rightx*param.xm_per_pix, 2)
   
  
   
    ## Visualization ##
    # Colors in the left and right lane regions
    fit_img = np.dstack((binary_warped, binary_warped, binary_warped))
    if param.debug:
        # Create an output image to draw on and visualize the result
        #fit_img = np.dstack((binary_warped, binary_warped, binary_warped))
        try:
            left_fitx = polyx(ploty, left_fit)
            right_fitx = polyx(ploty, right_fit)
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            left_fitx = 1*ploty**2 + 1*ploty
            right_fitx = 1*ploty**2 + 1*ploty
        
        fit_img[lefty, leftx] = [255, 0, 0]
        fit_img[righty, rightx] = [0, 0, 255]

        # Plots the left and right polynomials on the lane lines
        #plt.plot(left_fitx, ploty, color='yellow')
        #plt.plot(right_fitx, ploty, color='yellow')
        py = np.array(ploty, dtype=np.int32)
        lx = np.array(left_fitx + 0.5, dtype=np.int32)
        left_pts = np.vstack([lx, py]).T
        
        rx = np.array(right_fitx + 0.5, dtype=np.int32)
        right_pts = np.vstack([rx, py]).T
        cv2.polylines(fit_img, [left_pts, right_pts], False,  param.poly_line_color,  param.poly_line_thickness)   
        
    return left_fit, right_fit, left_fit_real, right_fit_real, ploty, fit_img
