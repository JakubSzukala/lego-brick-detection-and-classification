from roi_extractor import legoImageSegmentator 

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

import os 
import sys
from collections import namedtuple


if __name__ == '__main__':
    save_path_color = 'static_data/labeled_data/color_roi'
    save_path_binary = 'static_data/labeled_data/binary_roi'
    save_path_approx = 'static_data/labeled_data/approx_roi'
    # Declare set of valid labels for classification 
    valid_labels = [str(l) for l in range(0, 6)]
    print('Valid labels: ', valid_labels)

    # Declare class counter with count of zero for each class
    classcnt = {
            'C0': 0,
            'C1': 0,
            'C2': 0,
            'C3': 0,
            'C4': 0,
            'C5': 0
            }

    for f in os.listdir('train_img'):
        l = legoImageSegmentator(os.path.join('train_img', f), scale=0.2)
        
        for segment in l:
            fig, axs = plt.subplots(2)
            fig.suptitle('roi')
            axs[0].imshow(segment.roi_color)
            axs[1].imshow(segment.roi_contour_bin, cmap=plt.cm.gray) 
            plt.show()
            
            c = 'dummy'
            while c not in valid_labels:
                c = input('Declare a class for given lego.')
            
            print('Picked class: ', c) 
            key = 'C' + c
            classcnt[key] += 1
            cv.imwrite(os.path.join(save_path_color, 
                key + '-' + str(classcnt[key]) + '.png'),
                segment.roi_color)
            cv.imwrite(os.path.join(save_path_binary, 
                key + '-' + str(classcnt[key]) + '.png'),
                segment.roi_contour_bin)
            cv.imwrite(os.path.join(save_path_approx, 
                key + '-' + str(classcnt[key]) + '.png'),
                segment.roi_approxed_shape)
            















