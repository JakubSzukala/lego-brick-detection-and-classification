import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('static_data/orb-classiffier-templates/1-1.png')

def do_notihg(value):
    pass


cv.namedWindow('asdf')
cv.createTrackbar('edgeThreshold', 'asdf', 0, 31, do_notihg)
cv.createTrackbar('patchSize', 'asdf', 2, 31, do_notihg)

# Initiate ORB detector
key = ord('a')
while key != ord('q'):
    eth = cv.getTrackbarPos('edgeThreshold', 'asdf')
    patch = cv.getTrackbarPos('patchSize', 'asdf')
    sift = cv.SIFT_create(nfeatures=1000, contrastThreshold=0.001)
    kp = sift .detect(img,None)
    kp, des = sift.compute(img, kp)
    img2 = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
    cv.imshow('asdf', img2)
    key = cv.waitKey(1)


# draw only keypoints location,not size and orientation
#img2 = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
#plt.imshow(img2), plt.show()
