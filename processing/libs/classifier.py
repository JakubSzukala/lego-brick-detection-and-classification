import cv2 as cv
import numpy as np

import os
import math

from processing.utility_scripts.pdf_generator import PDFFitter 
from .roi_extractor import legoImageROI

def vote(new_vote, votes_dict, importance=1.0):
    for key in votes_dict:
        votes_dict[key] += new_vote[key] * importance
    return votes_dict


def vote_color(color_votes):
    sorted_dict = sorted(((v, k) for k, v in color_votes.items()))
    first_color = sorted_dict[-1]
    second_color = sorted_dict[-2]
    if second_color[0] == 0.0:
        return first_color[-1] # color key
    else:
        return 'mixed'


def find_biggest_area(rois: list):
    # TODO THIS IS VERY VERY VERY BAD, CHANGE THIS 
    #biggest_area_roi = max(rois, key=lambda n : n.roi_cont_area) 
    areas = []
    for roi in rois:
        areas.append(roi.roi_cont_area)
    areas.sort()
    print(areas)
    idx = -1 
    while (areas[idx] - areas[idx - 1]) / areas[idx] > 0.185:
        idx -= 1
    
    # This is really bad... bandage fix, improve this 
    allowed_area = areas[idx]

    for i, roi in enumerate(rois):
        if roi.roi_cont_area > allowed_area:
            rois.pop(i)

    return areas[idx], rois


def is_noise(roi: legoImageROI, biggest_area: float, coeff: float):
    noise_lower_area_limit = biggest_area * coeff
    return roi.roi_cont_area < noise_lower_area_limit


# Matching shpaes
# Corners
# Histograms 
# Ratio of edges of rois

    
"""
Classifier
Classifier.init_static_data(pdfs, form of dict used for voting)
Classifier.classify(lego roi) -> dict with votes for each class
vote func taking such dict and summing scores
"""

class ROIShapeClassifier:
    """
    Assign votes with confidence based on the probability density function 
    created from training set. votes_dict and files under static_data_path
    must have the same designators (votes_dict keys must be C<class num> and
    for static_data_path filenames must be in form C<class num>_whatever.csv)
    """
    def __init__(self, votes_dict_keys: list, static_data_path: str, 
            pdf_epsilon: float):
        self.votes_dict = {key : 0.0 for key in votes_dict_keys}
        self.pdf_fitters = []
        self.pdf_epsilon = pdf_epsilon

        self.init_static_data(static_data_path)


    def init_static_data(self, data_filepath):
        static_data_files = os.listdir(data_filepath)
        static_data_files.sort()
        for f in static_data_files:
            if not f.endswith('shape.csv'):
                continue
            designator = f.split('_')[0]
            path = os.path.join(data_filepath, f)
            fitter = PDFFitter(path, designator, 'ratio', 
                    epsilon=self.pdf_epsilon)
            self.pdf_fitters.append(fitter)
    

    def classify(self, roi: legoImageROI):
        h, w = roi.roi_color.shape[:2]
        if h > w:
            major = h
            minor = w
        else:
            major = w
            minor = h
        ratio = minor / major
        for pdf in self.pdf_fitters:
            self.votes_dict[pdf.designator] = round(pdf.query(ratio), 4)
        
        return self.votes_dict
    

    def check_if_noise(self, votes_dict, prcnt_th):
        """
        From observations it is easy to see that the ratio of edges of noise
        shapes is usually way outside the range of normal lego (trees are
        goo example). So comparing max result from this classifier to some 
        constant will indicate if given shape is noise or not.
        """
        max_val_key = max(votes_dict, key=votes_dict.get)
        max_val = votes_dict[max_val_key]
        return max_val < prcnt_th 
         

class ROIHistClassifier:
    def __init__(self, votes_dict_keys: list, static_data_path: str,
            pdf_epsilon):
        self.votes_dict = {key : 0.0 for key in votes_dict_keys}
        self.pdf_fitters = []
        self.pdf_epsilon = pdf_epsilon 

        self.init_static_data(static_data_path)
    
    # TODO: make it an inherited class
    def init_static_data(self, data_filepath):
        static_data_files = os.listdir(data_filepath)
        static_data_files.sort()
        for f in static_data_files:
            if not f.endswith('wpix_ratio.csv'):
                continue
            designator = f.split('_')[0]
            path = os.path.join(data_filepath, f)
            fitter = PDFFitter(path, designator, 'wpix_ratio', 
                    epsilon=self.pdf_epsilon)
            self.pdf_fitters.append(fitter)


    def classify(self, roi: legoImageROI):
        hist = cv.calcHist([roi.roi_approxed_shape], [0], None, [2], [0, 256])
        black_pix = float(hist[0])
        white_pix = float(hist[1])
        pix_sum = black_pix + white_pix
        wpix_ratio = white_pix / pix_sum

        for pdf in self.pdf_fitters:
            self.votes_dict[pdf.designator] = round(pdf.query(wpix_ratio), 4)

        return self.votes_dict


class ROIShapeMatchClassifier:
    """
    Desc
    """
    def __init__(self, votes_dict_keys: list, static_data_path: str):
        self.votes_dict = {key : 0.0 for key in votes_dict_keys}
        self.templates_dict = {}

        self.init_static_data(static_data_path)
   
    def get_contour(self, cimg):
        cimg = cv.copyMakeBorder(cimg, 2, 2, 2, 2,
            cv.BORDER_CONSTANT, value=0)
        # In case more than one cont found, take only biggest
        cimg_contours, _ = cv.findContours(cimg, 
                cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cimg_contours_sorted = sorted(cimg_contours, 
                key=lambda x: cv.contourArea(x))
        cimg_cont = cimg_contours_sorted[-1]
        return cimg_cont


    def init_static_data(self, data_filepath):
        static_data_files = os.listdir(data_filepath)
        static_data_files.sort()
        
        for f in static_data_files:
            designator = f.split('-')[0]
            path = os.path.join(data_filepath, f)
            template_img = cv.imread(path)
            template_img_gray = cv.cvtColor(template_img, cv.COLOR_BGR2GRAY)
            template_cont = self.get_contour(template_img_gray)
            
            if designator in self.templates_dict:
                self.templates_dict[designator].append(template_cont)
            else:
                self.templates_dict[designator] = [template_cont]
        
    
    def classify(self, roi: legoImageROI):
        contour = self.get_contour(roi.roi_approxed_shape)
        
        def sigmoid(x):
            return 2 * (1 / (1 + math.exp(-2* x))) - 1
        
        for key in self.templates_dict: # keys the same as in self.votes_dict
            results = []
            for template in self.templates_dict[key]:
                 results.append(cv.matchShapes(contour, template, 
                        cv.CONTOURS_MATCH_I3, None))
            average_similarity = np.mean(results)
            # Map with sigmoid to 0-1 range 
            # and invert it so bigger value = more similar 
            mapped_01 = sigmoid(average_similarity) 
            self.votes_dict[key] = round(1.0 - mapped_01, 4)

        return self.votes_dict

                
class ROIColorClassifier:
    def __init__(self):
        """
        Color ranges
        """
        # Move it outside the class?
        self.color_ranges = {}
        
        red_lower1 = np.array([0, 24, 0])
        red_upper1 = np.array([6, 255, 255])
        red_lower2 = np.array([171, 0, 0])
        red_upper2 = np.array([179, 255, 255]) 
        self.color_ranges['red'] = [
                (red_lower1, red_upper1),
                (red_lower2, red_upper2)
                ]
        
        blue_lower = np.array([97, 103, 0])
        blue_upper = np.array([120, 255, 255])
        self.color_ranges['blue'] = [(blue_lower, blue_upper)]

        green_lower = np.array([64, 50, 0])
        green_upper = np.array([89, 255, 255])
        self.color_ranges['green'] = [(green_lower, green_upper)]

        white_lower = np.array([0, 0, 0])
        white_upper = np.array([179, 89, 255])
        self.color_ranges['white'] = [(white_lower, white_upper)]

        yellow_lower = np.array([14, 109, 0])
        yellow_upper = np.array([55, 255, 255])
        self.color_ranges['yellow'] = [(yellow_lower, yellow_upper)]
        
    
    def get_mask_for_color(self, hsv_img, color):
        """
        Give the image with minimum area outside of interest.
        """
        h, w, _ = hsv_img.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        # Add multiple ranges masks together to create one
        for ranges in self.color_ranges[color]:
            next_mask = cv.inRange(hsv_img,
                    ranges[0], ranges[1])
            mask = cv.bitwise_or(mask, next_mask)
        
        return mask
    
    
    def calc_matched_color_ratio(self, mask):
        """
        White pixels match the color, black are masked and they don't.
        """
        hist = cv.calcHist([mask], [0], None, [2], [0, 256])
        black_pix = float(hist[0])
        white_pix = float(hist[1])
        pix_sum = black_pix + white_pix
        wpix_ratio = white_pix / pix_sum
        return wpix_ratio

    
    def classify_color(self, roi: legoImageROI):
        # Reduce the background with use of ready hull mask
        reduced_img = cv.bitwise_or(roi.roi_color, roi.roi_color, 
                mask=roi.roi_approxed_shape) 
        reduced_img = cv.cvtColor(reduced_img, cv.COLOR_BGR2HSV) 
        
        color_votes = {} 
        for color in self.color_ranges:
            mask = self.get_mask_for_color(reduced_img, color)
            # Remove little bit more bgnd
            mask = cv.bitwise_and(mask, roi.roi_approxed_shape)
            color_presence_prcnt = self.calc_matched_color_ratio(mask)
            color_votes[color] = color_presence_prcnt
        
        # Adjusting for white bgnd fragments, usually constitute ~20 - 30%  
        color_votes['white'] -= 0.3
        if color_votes['white'] < 0.0:
            color_votes['white'] = 0.0
         
        return color_votes






"""
Garbage
"""
"""
class templateClassifier:
    
    def load_static_data(self, path):
        templates = NUMBER_OF_CLASSES * [[]]
        for f in os.listdir(path):
            class_index = int(f.split('-')[0]) - 1 # first char
            
            # Prep template TODO: move V to legoImageProcessor?
            template = cv.imread(os.path.join(path, f), 0)
           
            # Load template to the list 
            templates[class_index].append(template)
        return tuple(templates)
     

class ORBClassifier(legoImageProcessor, templateClassifier):
    def __init__(self):
        self.orb = cv.ORB_create(edgeThreshold=15, nfeatures=100)
        self.bf_matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        self.templates = self.load_static_data(
                'static_data/orb-classiffier-templates') 
    
        
    def classify(self, lego_roi: legoRegionOfInterest):
        #ClassSimilarityScore = namedtuple('ClassSimilarityScore', 
        #        ['class', 'score'])
        img = lego_roi.brick_gray
        
        for c in self.templates:
            for template in c:
                # Find key points and descriptors
                kp_img, des_img = self.orb.detectAndCompute(img, None)
                kp_template, des_template = self.orb.detectAndCompute(
                        template, None)
                
                # Testing
                #temp = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
                cp = template.copy()
                img_kp = cv.drawKeypoints(cp, kp_template, None, 
                        color=(0, 150, 0), flags=0) 
                #plt.imshow(img_kp), plt.show()

                matches = self.bf_matcher.match(des_img, des_template)
                matches = sorted(matches, key=lambda x:x.distance)

                if False:
                    draw_img = cv.drawMatches(
                            img,kp_img,template,kp_template,matches[:10],None,
                            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                    print(len(matches))
                    plt.imshow(draw_img)
                    plt.show()


class SIFTClassifier(legoImageProcessor, templateClassifier):

    def __init__(self):
        self.sift = cv.SIFT_create(contrastThreshold=0.001)
        self.bf_matcher = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
        self.templates = self.load_static_data(
                'static_data/orb-classiffier-templates') 
            
     
    def classify(self, lego_roi: legoRegionOfInterest):
        img = lego_roi.brick_gray
        
        for c in self.templates:
            for template in c:
                # Find key points and descriptors
                kp_img, des_img = self.sift.detectAndCompute(img, None)
                kp_template, des_template = self.sift.detectAndCompute(
                        template, None)
                
                # Testing
                #temp = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
                cp = template.copy()
                img_kp = cv.drawKeypoints(cp, kp_template, None, 
                        color=(0, 150, 0), flags=0) 
                plt.imshow(img_kp), plt.show()
                
                matches = self.bf_matcher.match(des_img, des_template)
                matches = sorted(matches, key=lambda x:x.distance)

                if False:
                    draw_img = cv.drawMatches(
                            img,kp_img,template,kp_template,matches[:10],None,
                            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                    print(len(matches))
                    plt.imshow(draw_img)
                    plt.show()
"""
                

         



