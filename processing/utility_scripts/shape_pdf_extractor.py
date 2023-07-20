import os
import sys
import csv 
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


# TODO: move this functions to some convinient place so they do not have to 
# be redefined in classifiers!!!
def shape_extract(img):
    h, w = img.shape[:2]

    if h > w:
        major = h
        minor = w
    else:
        major = w
        minor = h

    ratio = round(minor / major, 4)
    return [h, w, ratio]


def white_pix_percent_extract(img):
    hist = cv.calcHist([img], [0], None, [2], [0, 256])
    black_pix = float(hist[0])
    white_pix = float(hist[1])
    pix_sum = black_pix + white_pix
    wpix_ratio = white_pix / pix_sum
    return [black_pix, white_pix, wpix_ratio]
    

class featureExtractor:
    """
    Class for extracting relevant informations from an image and save it to
    the file. 
    """
    def __init__(self, output_data_path, output_file_suffix, classes_num,
            data_cols, ignored_classes = [None]):
        self.output_data_path = output_data_path 
        self.output_file_suffix = output_file_suffix
        self.classes_num = classes_num
        self.data_cols = data_cols
        
        self.file_writers = self.generate_csv_writers() 

   
    def write2csv(self, designator, img, extractor_func):
        data_row = extractor_func(img)
        self.file_writers[designator][0].writerow(data_row)


    def generate_csv_writers(self):
        file_writers = self.generate_designators_dict(self.classes_num)
        for key in file_writers:
            filename = key + '_' + self.output_file_suffix + '.csv' 

            # Create a class file / writer tuple 
            class_file = open(os.path.join(
                self.output_data_path, filename), 'w')
            writer = csv.writer(class_file, delimiter=',', 
                quotechar='|', quoting=csv.QUOTE_MINIMAL) 
            writer.writerow(self.data_cols)
            file_writers[key] = (writer, class_file)
        return file_writers
    

    def extract_designator(self, filename):
            return filename.split('-')[0]# first digit / up to '-' is class


    def num2idx(self, class_num):
        return class_num - 1


    def generate_designators_dict(self, classes_num):
        dd = {}
        for i in range(1, classes_num + 1):
            dd['C{}'.format(i)] = None
        return dd

    
    def close_files(self):
        for key in self.file_writers:
            self.file_writers[key][1].close()


if __name__ == '__main__':
    path = '../static_data/labeled_data/approx_roi'
    files = os.listdir(path)
    files.sort()

    shape_extractor = featureExtractor('../static_data/data/', 'shape', 5, 
            ['h', 'w', 'ratio'])
    wpix_ratio_extractor = featureExtractor('../static_data/data/', 
            'wpix_ratio', 5, ['bpix', 'wpix', 'wpix_ratio'])
    for f in files:
        designator = f.split('-')[0]

        # Ignore noise class
        if designator == 'C0':
            continue
        
        filepath = os.path.join(path, f)
        shape_extractor.write2csv(designator, 
                cv.imread(filepath, 0), shape_extract)
        wpix_ratio_extractor.write2csv(designator, 
                cv.imread(filepath, 0), white_pix_percent_extract)


    shape_extractor.close_files()


"""
for f in files:
    lego_class = int(f.split('-')[0].lstrip('0').strip('C'))# first digit / up to '-' is class
    lp = int(f.split('-')[1].split('.')[0])
    
    # Do not include noise class
    if lego_class == 0:
        continue

    lego = cv.imread(os.path.join(path, f))
    h, w = lego.shape[:2]

    if h > w:
        major = h
        minor = w
    else:
        major = w
        minor = h

    ratio = round(minor / major, 4)
     
    file_writers[num2idx(lego_class)][0].writerow([lp, h, w, ratio]) 

for w in file_writers:
    w[1].close()

"""











