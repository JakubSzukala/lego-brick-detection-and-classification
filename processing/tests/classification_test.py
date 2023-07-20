"""
Resolving a problem with importing packages from parent directory:
https://stackoverflow.com/questions/11536764/how-to-fix-attempted-relative-import-in-non-package-even-with-init-py/27876800#27876800
"""
import sys
import os
from collections import namedtuple
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classifier import ROIShapeClassifier
from classifier import ROIHistClassifier
from classifier import ROIShapeMatchClassifier
from classifier import vote
from classifier import is_noise, find_biggest_area
from  roi_extractor import legoImageROI
from main import print_hs # TODO: move this to some utils or smth

import cv2 as cv
import termcolor
from matplotlib import pyplot as plt

class Score:
    def __init__(self, name):
        self.name = name
        self.correct = 0
        self.incorrect = 0
        self.success_rate = 0.0 


    def add_result(self, is_correct: bool):
        if is_correct:
            self.correct += 1
        else:
            self.incorrect += 1
        self._calc_ratio()

    
    def _calc_ratio(self):
        results_sum = self.correct + self.incorrect 
        self.success_rate = float(self.correct) / float(results_sum)
    

    def __repr__(self):
        ret_val = \
                ' {} '.format(self.name) + \
                termcolor.colored('Correct: {} '.format(self.correct),
                        'green') + \
                termcolor.colored('Incorrect: {} '.format(self.incorrect), 
                        'red') + \
                'Success Rate: {}'.format(round(self.success_rate, 2))

        return ret_val


def load_labeled_data():
    """
    Labeled data is saved in 3 dirs:
    * approx_roi
    * binary_roi
    * color_roi
    Each classification subject has a 3 images, one in each directory 
    and their names are exactly the same but differ in path.
    """
    APPROX_ROI_PATH = '../static_data/labeled_data/approx_roi'
    BINARY_ROI_PATH = '../static_data/labeled_data/binary_roi'
    COLOR_ROI_PATH = '../static_data/labeled_data/color_roi'
    
    rois = []
    for name in os.listdir(APPROX_ROI_PATH):
        label = name.split('-')[0] 
        approx_roi = cv.imread(os.path.join(APPROX_ROI_PATH, name))
        approx_roi = cv.cvtColor(approx_roi, cv.COLOR_BGR2GRAY)
        binary_roi = cv.imread(os.path.join(BINARY_ROI_PATH, name))
        color_roi = cv.imread(os.path.join(COLOR_ROI_PATH, name))
        
        contours, _ = cv.findContours(approx_roi, # TODO: this is not great...
                    cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours_sorted = sorted(contours, 
                key=lambda x: cv.contourArea(x))
        contour = contours_sorted[-1]
        roi_contour_area = cv.contourArea(contour)

        new_lego_roi = legoImageROI(color_roi, binary_roi, approx_roi, 
                roi_contour_area)

        rois.append((new_lego_roi, label)) 
    
    return rois 


if __name__ == '__main__':
    # TODO: fix this dirty hack
    rois_labels = load_labeled_data()
    
    votes_dict_keys = ['C1', 'C2', 'C3', 'C4', 'C5'] 
    shape_classifier = ROIShapeClassifier(
            votes_dict_keys, '../static_data/data', pdf_epsilon=0.03)
    hist_classifier = ROIHistClassifier(
            votes_dict_keys, '../static_data/data', pdf_epsilon=0.03)
    shape_match_classifier = ROIShapeMatchClassifier(
            votes_dict_keys, '../static_data/shape_templates')
   
    score = Score('General Score')
    score_shape = Score('Shape Classifier Score')
    score_hist = Score('Histogram Classifier Score')
    score_shape_match = Score('Shape Match Classifier Score')
    
    rois = list(zip(*rois_labels))[0]
    biggest_area = find_biggest_area(rois) # type: ignore

    for roi, label in rois_labels:
        # This does not make sense, no distinction on images = different size
        #if is_noise(roi, biggest_area, coeff=0.6): 
        #    print('Classified as noise: ', label)
        #    continue
        if label == 'C0':
            continue

        votes_dict = {
            'C1' : 0.0,
            'C2' : 0.0,
            'C3' : 0.0,
            'C4' : 0.0,
            'C5' : 0.0
        }
        
        shape_vote = shape_classifier.classify(roi)
        hist_vote = hist_classifier.classify(roi)
        shape_match_vote = shape_match_classifier.classify(roi)
        
        vote(shape_vote, votes_dict, importance=0.5)
        vote(hist_vote, votes_dict, importance=0.25)
        vote(shape_match_vote, votes_dict)
            
        # TODO: this is pretty ugly, improve this V
        classification_result = max(votes_dict, key=votes_dict.get)
        shape_vote_cr = max(shape_vote, key=shape_vote.get)
        hist_vote_cr = max(hist_vote, key=hist_vote.get)
        shape_match_vote_cr = max(shape_match_vote, key=shape_match_vote.get)
        
        is_correct = label == classification_result
        score.add_result(is_correct)
        score_shape.add_result(label == shape_vote_cr)
        score_hist.add_result(label == hist_vote_cr)
        score_shape_match.add_result(label == shape_match_vote_cr)

        
        if not is_correct: 
            # Print
            title = 'Identified as: ' + classification_result + \
                ' and label is: ' + label 
            print('##############################') 
            print(title)
            print(label, classification_result)
            print('Shape classifier: ')
            print_hs(shape_vote) 
            print('Histogram classifier: ')
            print_hs(hist_vote)
            print('Shape match classifier: ')
            print_hs(shape_match_vote)
            print('Sum / final score: ')
            print_hs(votes_dict)
            
                        
            # Plot 
            fig, axs = plt.subplots(1, 3)
            fig.suptitle(title)
            axs[0].imshow(roi.roi_color)
            axs[1].imshow(roi.roi_contour_bin, cmap=plt.cm.gray) 
            axs[2].imshow(roi.roi_approxed_shape, cmap=plt.cm.gray) 
            plt.show()
    print(score_shape)
    print(score_hist)
    print(score_shape_match)
    print(score)




        












