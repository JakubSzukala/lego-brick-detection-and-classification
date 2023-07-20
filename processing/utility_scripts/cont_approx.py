import cv2 as cv
import numpy as np
import sys

def dummy(val):
    pass

# make window
cv.namedWindow('trackbar')

cv.createTrackbar('scale', 'trackbar', 10, 100,
        dummy)

# display in loop
path = sys.argv[1]
img_color = cv.imread(path)
img_color = cv.copyMakeBorder(img_color, 10, 10, 10 ,10,
        cv.BORDER_CONSTANT, value=(0, 0, 0))
img = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)
img = cv.resize(img, (0, 0), fx=5.0, fy=5.0)
img_color = cv.resize(img_color, (0, 0), fx=5.0, fy=5.0)
_, img = cv.threshold(img, 10, 255, cv.THRESH_BINARY)
print(img.shape)
print(img.dtype)
c, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
print(c[0].dtype)
#cv.drawContours(img_color, c, -1, (0, 150, 0), thickness=3)
#cv.imshow('color', img_color)
#cv.waitKey()

hull = []

# calculate points for each contour
for i in range(len(c)):
    # creating convex hull object for each contour
    hull.append(cv.convexHull(c[i], False))

# create an empty black image
drawing = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)

# draw contours and hull points
for i in range(len(c)):
    color_contours = (0, 255, 0) # green - color for contours
    color = (255, 0, 0) # blue - color for convex hull
    # draw ith contour
    cv.drawContours(drawing, c, i, color_contours, 1, 8)
    # draw ith convex hull object
    cv.drawContours(drawing, hull, i, color, 1, 8)

cv.imshow('asdf',drawing)
cv.waitKey()

"""
key = ord('a')
while key != ord('q'):
    gimg = img.copy()
    cimg = img_color.copy()
    scale = float(cv.getTrackbarPos('scale', 'trackbar')) / 100.0
    approxed = None
    for cnt in c:
        epsilon = scale * cv.arcLength(cnt, True)
        approxed = cv.approxPolyDP(cnt, epsilon, True)
        print(approxed)
        cv.drawContours(cimg, approxed, -1, (0, 150, 0), 3)
    print('trackbar: ', scale)
    cv.imshow('trackbar', cimg)
    key = cv.waitKey(5)
"""
