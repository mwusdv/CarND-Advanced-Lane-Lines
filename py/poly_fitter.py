import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2

from lane_detect_param import LaneDetectParam


class PolyFitter:
    def __init__(self, param):
        self._param = param
        self.clear()
        
    def clear(self):
        self._left_fit = None
        self._right_fit = None
        
        self._left_fit_real = None
        self._right_fit_real = None
        
        self._leftx = None
        self._lefty = None
        self._rightx = None
        self._righty = None
        
    def find_lane_pixels(self, binary_warped):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
         
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0]//self._param.hist_nwindows)
        
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
        for window in range(self._param.hist_nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            
            win_xleft_low = leftx_current - self._param.hist_margin  
            win_xleft_high = leftx_current+ self._param.hist_margin
            win_xright_low = rightx_current - self._param.hist_margin
            win_xright_high = rightx_current+ self._param.hist_margin
            
            # Draw the windows on the visualization image
            #if param.debug:
            #    cv2.rectangle(out_img,(win_xleft_low,win_y_low),
            #    (win_xleft_high,win_y_high),(0,255,0), 2) 
            #    cv2.rectangle(out_img,(win_xright_low,win_y_low),
            #    (win_xright_high,win_y_high),(0,255,0), 2) 
                
            # Identify the nonzero pixels in x and y within the window 
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                              (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            
            
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            # update left and right center
            if good_left_inds.shape[0] > self._param.hist_minpix:
                leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
            
            if good_right_inds.shape[0] > self._param.hist_minpix:
                rightx_current = np.int32(np.mean(nonzerox[good_right_inds])) 
                
        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    
        # Extract left and right line pixel positions
        self._leftx = nonzerox[left_lane_inds]
        self._lefty = nonzeroy[left_lane_inds] 
        self._rightx = nonzerox[right_lane_inds]
        self._righty = nonzeroy[right_lane_inds]
        
    def search_around_poly(self, binary_warped):
        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        #Set the area of search based on activated x-values
        leftx = polyx(nonzeroy, left_fit)
        left_lane_inds = ((nonzerox > leftx-margin) & (nonzerox < leftx+margin))
        
        rightx = polyx(nonzeroy, right_fit)
        right_lane_inds = ((nonzerox > rightx-margin) & (nonzerox < rightx+margin))
        
        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
    
        # Fit new polynomials
        left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
        
        ## Visualization ##
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                                  ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                                  ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))
    
        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        
        # Plot the polynomial lines onto the image
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        ## End visualization steps ##
        
        return result

    def fit_polynomial(binary_warped):
        # Find our lane pixels first
        leftx, lefty, rightx, righty = find_lane_pixels(binary_warped, self._param)
    
        #Fit a second order polynomial
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
    
        left_fit_real = np.polyfit(lefty*self._param.ym_per_pix, leftx*self._param.xm_per_pix, 2)
        right_fit_real = np.polyfit(righty*self._param.ym_per_pix, rightx*self._param.xm_per_pix, 2)
       
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
       
        ## Visualization ##
        # Colors in the left and right lane regions
        fit_img = None
        if self._param.debug:
            # Create an output image to draw on and visualize the result
            fit_img = np.dstack((binary_warped, binary_warped, binary_warped))
            try:
                left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
                right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
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
            cv2.polylines(fit_img, [left_pts, right_pts], False,  self._param.poly_line_color,  self._param.poly_line_thickness)   
            
        return left_fit, right_fit, left_fit_real, right_fit_real, ploty, fit_img
