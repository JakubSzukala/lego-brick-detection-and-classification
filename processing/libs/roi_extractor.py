import cv2 as cv
import numpy as np

from matplotlib import pyplot as plt

import os
from collections import namedtuple

# TODO
# * make processing into func, so no new obj have to be created for each img
"""
##############################################################################
Class definitions 
##############################################################################
"""

"""
Named tuple for storing and passing around rois.
"""
legoImageROI = namedtuple('legoImageROI', 
        [
            'roi_color', 
            'roi_contour_bin', 
            'roi_approxed_shape',
            'roi_cont_area'
            ])


class legoImageSegmentator:
    
    def __init__(self, img, scale = 0.3):
        """
        BEGIN Parameters for image preprocessing 
        """
        # Gaussian HPS params
        SIGMA = 1                                    
        
        # HSV masking params
        HSV_LOWER_LIMIT = np.array([0, 80, 0])       # HSV lower range limit
        HSV_UPPER_LIMIT = np.array([179, 255, 255])  # HSV upper range limit
        
        # Canny edges params 
        KERNEL = np.ones((3, 3), np.uint8)           
        TH1 = 34                                     
        TH2 = 60

        # Smallest contour 
        SMALLEST_CONT = 40
        """
        END Parameters for image preprocessing
        """
        
        # Load image, rescale if necessary 
        self.original_image = img
        self.scale = scale
        if self.scale != 1.0: # dangerous float comparission...
            self.original_image = cv.resize(self.original_image, (0, 0),
                    fx=self.scale, fy=self.scale)
            print('Rescaled image to scale: ', self.scale, 'of original size')
        
        # Usefull versions of the image
        self.gray_img = cv.cvtColor(self.original_image, cv.COLOR_BGR2GRAY)
        self.hsv_img = cv.cvtColor(self.original_image, cv.COLOR_BGR2HSV)

        # Images at various preprocessing steps
        self.filtered_gauss_hps_img = self.gauss_hps(self.gray_img, 
                sigma=SIGMA)
        self.hsv_mask_img = self.get_hsv_mask(self.hsv_img, 
                lower=HSV_LOWER_LIMIT, upper=HSV_UPPER_LIMIT) 
        self.canny_edges_img = self.get_canny_edges(self.gray_img, 
                th1=TH1, th2=TH2, kernel=KERNEL)
        self.hsv_and_canny_img = cv.add(
                self.canny_edges_img, self.hsv_mask_img) 
        
        # Contours / bboxes
        self.contours = self.find_contours(self.hsv_and_canny_img)
        cv.fillPoly(self.hsv_and_canny_img, self.contours, 255, 1, 0)
        
        # TODO: rearrange in more logical way
        self.approxed_shapes = self.hsv_and_canny_img.copy()
        self.temp = np.zeros((self.original_image.shape[0],
            self.original_image.shape[1], 1), np.uint8)
        self.bboxes_img = self.original_image.copy() 
        
        # Create list of legoImageROIs
        self.lego_rois = []
        for contour in self.contours:
            if len(contour) < SMALLEST_CONT:
                continue
            bbox = cv.minAreaRect(contour)
            
            # Shape approx for particular contour
            hull = cv.convexHull(contour, False)
            cv.fillPoly(self.approxed_shapes, [hull], 255, 8, 0)

            roi_color = self.get_roi_from_bbox(
                    bbox, self.original_image)
            roi_contour_bin = self.get_roi_from_bbox(
                    bbox, self.hsv_and_canny_img)
            roi_approxed_no_convx = self.get_roi_from_bbox(
                    bbox, self.approxed_shapes)
            roi_contour_area = self.get_cont_area(roi_approxed_no_convx)

            roi = legoImageROI(
                    roi_color,
                    roi_contour_bin,
                    roi_approxed_no_convx,
                    roi_contour_area)
            
            self.lego_rois.append(roi) 
            
            # Drawing
            points = cv.boxPoints(bbox).astype(np.int0)
            cv.drawContours(self.bboxes_img, [points], 0, (0, 150, 0),
                    thickness=3) 

        if False:
            fig, axs = plt.subplots(2, 3)
            fig.suptitle('All relevant image processing steps.')
            axs[0, 0].imshow(self.original_image)
            axs[0, 0].set_title('Original image')

            axs[0, 1].imshow(self.filtered_gauss_hps_img)
            axs[0, 1].set_title('Highpass filter')
            
            axs[0, 2].imshow(self.hsv_mask_img) # type: ignore
            axs[0, 2].set_title('Hsv masked colors (except white)')

            axs[1, 0].imshow(self.canny_edges_img)
            axs[1, 0].set_title('Canny edges')
            
            axs[1, 1].imshow(self.hsv_and_canny_img)
            axs[1, 1].set_title('Sum of hsv mask and canny edges' \
                    'with connected poly')
            
            axs[1, 2].imshow(self.bboxes_img)
            axs[1, 2].set_title('Bounding boxes of detected shapes')
            
            plt.show()
    
    def segment_lego_image(self):
        return NotImplementedError('Hi, implement me pls.')
    

    def gauss_hps(self, gray_img, sigma):
        return gray_img - cv.GaussianBlur(gray_img, (21, 21), sigma) + 127


    def get_hsv_mask(self, hsv_color_img, lower, upper):
        return cv.inRange(hsv_color_img, lower, upper)


    def get_canny_edges(self, gray_img, th1, th2, kernel):
        canny_img = cv.Canny(gray_img, th1, th2, L2gradient=True)
        
        improved_canny_close = cv.morphologyEx(canny_img, 
                cv.MORPH_CLOSE, kernel, iterations=2)
        improved_canny_dilated = cv.morphologyEx(improved_canny_close, 
                cv.MORPH_DILATE, kernel, iterations=1)
        return improved_canny_dilated

    """
    Iterator class for easier operation on extracted rois.
    """
    class roiIterator:
        
        def __init__(self, lego_img_segmentator):
            self.lego_img_segmentator = lego_img_segmentator 
            self.idx = -1
            self.end_idx = len(lego_img_segmentator.lego_rois) - 1


        def __iter__(self):
            return self
       

        def __next__(self):
            self.idx += 1
            if self.idx > self.end_idx:
                raise StopIteration
            return self.lego_img_segmentator.lego_rois[self.idx]


    def __iter__(self):
        return self.roiIterator(self)
    
       
    def find_contours(self, bin_img):
        contours, _ = cv.findContours(bin_img, 
                cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        return contours
        
    
    def get_roi_from_bbox(self, bbox, original_image): 
        # https://stackoverflow.com/questions/11627362
        # Get minimum area rect from a contour 
        
        # Extract data into understandable variables and validate
        center = bbox[0]
        width, height = bbox[1]
        height = int(height)
        width = int(width)
        angle = bbox[2]

        # Rotate entire original_image by given angle so slicing is possible
        M = cv.getRotationMatrix2D(center, angle, 1)
        
        # This convention where u give dims of img as (x, y)!!!
        rot_img_shape = \
                (original_image.shape[1], # x
                original_image.shape[0])  # y
        
        rot_img = cv.warpAffine(original_image, M, rot_img_shape)
        
        # Np slice image to get the roi
        x = int(center[0] - width/2)
        y = int(center[1] - height/2)
        
        # Avoid negative indexes, as they count from the end
        if x < 0: x = 0
        if y < 0: y = 0

        out = rot_img[y : y + height, x : x + width] 
        return out
    
    
    def get_cont_area(self, approxed_bin_img):
        contours, _ = cv.findContours(approxed_bin_img,
                cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        # Eliminate potential false positive contours
        contours_sorted = sorted(contours, 
                key=lambda x: cv.contourArea(x))
        contour = contours_sorted[-1]

        return cv.contourArea(contour)















