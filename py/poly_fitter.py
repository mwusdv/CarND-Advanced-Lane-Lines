import numpy as np
import cv2

def compute_curvature(poly2_fit, y):
    a, b, c = poly2_fit
    curvature = (1+(2*a*y+b)**2)**1.5/2/np.abs(a)
    return curvature

def polyx(y, poly_fit):
    return poly_fit[0] * y**2 + poly_fit[1] * y + poly_fit[2]

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
        
        self._fit_img = None
        self._ploty = None
        
    # Tind possible lane pixels
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
        
    # Find possible lane pixels around poly curves
    def search_around_poly(self, binary_warped):
        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # Set the area of search based on activated x-values
        leftx = polyx(nonzeroy, self._left_fit)
        left_lane_inds = ((nonzerox > leftx-self._param.hist_margin) & (nonzerox < leftx+self._param.hist_margin))
        
        rightx = polyx(nonzeroy, self._right_fit)
        right_lane_inds = ((nonzerox > rightx-self._param.hist_margin) & (nonzerox < rightx+self._param.hist_margin))
        
        # Extract left and right line pixel positions
        self._leftx = nonzerox[left_lane_inds]
        self._lefty = nonzeroy[left_lane_inds] 
        self._rightx = nonzerox[right_lane_inds]
        self._righty = nonzeroy[right_lane_inds]
   
    def fit_gap_small(self, fit1, fit2):
        gap = abs(fit1 - fit2)
        return np.max(gap) < self._param.fit_gap_ub
    
    # Fuse newly estimated poly with the existing one 
    def update_fit(self, left_fit, right_fit, left_fit_real, right_fit_real):
        if self._left_fit is not None and self._param.consider_prev:
            if self.fit_gap_small(left_fit, self._left_fit):
                self._left_fit = self._left_fit*self._param.fit_momentum + left_fit*(1-self._param.fit_momentum)
                self._left_fit_real = self._left_fit_real*self._param.fit_momentum + left_fit_real*(1-self._param.fit_momentum)
            
            if self.fit_gap_small(right_fit, self._right_fit):
                self._right_fit = self._right_fit*self._param.fit_momentum + right_fit*(1-self._param.fit_momentum)
                self._right_fit_real = self._right_fit_real*self._param.fit_momentum + right_fit_real*(1-self._param.fit_momentum)
                
        else:
            self._left_fit = left_fit
            self._right_fit = right_fit
            self._left_fit_real = left_fit_real
            self._right_fit_real = right_fit_real
              
    # Fit poly curve given the possible lane pixels
    def fit_polynomial(self, binary_warped):
        # Find our lane pixels first
        if self._left_fit is not None and self._param.consider_prev:
            self.search_around_poly(binary_warped)
        else:
            self.find_lane_pixels(binary_warped)
            
        #Fit a second order polynomial
        if len(self._lefty) > 3:
            left_fit = np.polyfit(self._lefty, self._leftx, 2)
            left_fit_real = np.polyfit(self._lefty*self._param.ym_per_pix, self._leftx*self._param.xm_per_pix, 2)
        else:
            left_fit = self._left_fit
            left_fit_real = self._left_fit_real
            
        if len(self._righty) > 3:
            right_fit = np.polyfit(self._righty, self._rightx, 2)
            right_fit_real = np.polyfit(self._righty*self._param.ym_per_pix, self._rightx*self._param.xm_per_pix, 2)
        else:
            right_fit = self._right_fit
            right_fit_real = self._right_fit_real
    
        # Fuse newly estimated poly with the existing one 
        self.update_fit(left_fit, right_fit, left_fit_real, right_fit_real)
        
        # Generate x and y values for plotting
        self._ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
       
        # points on the fitted curves    
        self._left_fitx = self._left_fit[0]*self._ploty**2 + self._left_fit[1]*self._ploty + self._left_fit[2]
        self._right_fitx = self._right_fit[0]*self._ploty**2 + self._right_fit[1]*self._ploty + self._right_fit[2]      
     
        py = np.array(self._ploty, dtype=np.int32)  
        
        lx = np.array(self._left_fitx + 0.5, dtype=np.int32) # add 0.5 in order to cast to integer more accurately
        left_pts = np.vstack([lx, py]).T  
        
        # only keep the non-negative points
        idx = lx >= 0
        self._left_pts = left_pts[idx, :]
        
        rx = np.array(self._right_fitx + 0.5, dtype=np.int32)
        right_pts = np.vstack([rx, py]).T
        
        # only keep the non-negative points
        idx = rx >= 0
        self._right_pts = right_pts[idx, :]
        
        # Create an output image to draw on and visualize the result
        if self._param.debug:
            self._fit_img = np.dstack((binary_warped, binary_warped, binary_warped))
            self._fit_img[self._lefty, self._leftx] = [255, 0, 0]
            self._fit_img[self._righty, self._rightx] = [0, 0, 255]
            cv2.polylines(self._fit_img, [self._left_pts, self._right_pts], False,  self._param.poly_line_color,  self._param.poly_line_thickness)   
        
        # compute curvature
        self.measure_curvature_real()
        
        return self._fit_img
        
        
    # Calculates the curvature of polynomial functions in meters.          
    def measure_curvature_real(self):
        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(self._ploty)
        
        # calculation of R_curve (radius of curvature)
        self._left_curverad = compute_curvature(self._left_fit_real, y_eval*self._param.ym_per_pix)
        self._right_curverad = compute_curvature(self._right_fit_real, y_eval*self._param.ym_per_pix)
         