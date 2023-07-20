from typing import List

import numpy as np

from processing.libs.roi_extractor import legoImageSegmentator
from processing.libs.classifier import ROIShapeClassifier
from processing.libs.classifier import ROIHistClassifier
from processing.libs.classifier import ROIShapeMatchClassifier
from processing.libs.classifier import ROIColorClassifier
from processing.libs.classifier import vote
from processing.libs.classifier import find_biggest_area
from processing.libs.classifier import is_noise
from processing.libs.classifier import vote_color

MappingClass2Idx = {
        'C1':0,
        'C2':1,
        'C3':2,
        'C4':3,
        'C5':4,
        'red':5,
        'green':6,
        'blue':7,
        'white':8,
        'yellow':9,
        'mixed':10
        }

votes_dict_keys = ['C1', 'C2', 'C3', 'C4', 'C5'] 
shape_classifier = ROIShapeClassifier(
        votes_dict_keys, 'processing/static_data/data', 0.03)
hist_classifier = ROIHistClassifier(
        votes_dict_keys, 'processing/static_data/data', 0.03)
shape_match_classifier = ROIShapeMatchClassifier(
        votes_dict_keys, 'processing/static_data/shape_templates')
color_classifier = ROIColorClassifier()

def perform_processing(image: np.ndarray) -> List[int]:
    print(f'image.shape: {image.shape}')
    l = legoImageSegmentator(image, scale=0.2)
    biggest_area, adjusted_rois_by_area = find_biggest_area(l.lego_rois)
    l.lego_rois = adjusted_rois_by_area

    json_vote = [0] * 11

    for segment in l:
        classified_as_noise = is_noise(segment, biggest_area, 0.625)
        if classified_as_noise:
            continue

        votes_dict = {
                'C1' : 0.0,
                'C2' : 0.0,
                'C3' : 0.0,
                'C4' : 0.0,
                'C5' : 0.0
                }

        shape_vote = shape_classifier.classify(segment)
        hist_vote = hist_classifier.classify(segment)
        shape_match_vote = shape_match_classifier.classify(segment)
        color_vote =  color_classifier.classify_color(segment)
        
        # This classifier is "great" in checking if roi is noise
        classified_as_noise = shape_classifier.check_if_noise(
                shape_vote, prcnt_th=0.05) # not greatest, marginal val
        if classified_as_noise:
            continue

        vote(shape_vote, votes_dict, importance=0.5)
        vote(hist_vote, votes_dict, importance=0.25)
        vote(shape_match_vote, votes_dict)
        final_verdict_class = max(votes_dict, key=votes_dict.get) 
        
        json_vote[MappingClass2Idx[final_verdict_class]] += 1
        json_vote[MappingClass2Idx[vote_color(color_vote)]] += 1

    return json_vote
