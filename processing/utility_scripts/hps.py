import cv2 as cv
import numpy as np
from skimage import exposure

def dummy(val):
    pass

def highpass1(img, sigma):
    p2, p98 = np.percentile(img, (2, 98))
    #img = exposure.rescale_intensity(img, in_range=(p2, p98)) # type: ignore

    g = cv.GaussianBlur(img, (21, 21), sigma)
    out = img - cv.GaussianBlur(g, (21, 21), sigma) + 127

    
    cv.imshow('gauss', out)
    #out = cv.Canny(out, 10, 10)
    #cv.imshow('canned', out)
    #cv.waitKey()
    return out

def highpass2(img):
    kernel = np.array([  [-1, -1, -1, -1, -1],
                        [-1, 1, 2, 1, -1],
                        [-1, 2, 8, 2, -1],
                        [-1, 1, 2, 1, -1],
                        [-1, -1, -1, -1, -1]])
    kernel = np.array([[-1, -1, -1],
                   [-1,  4, -1],
                   [-1, -1, -1]])
    return img - cv.filter2D(img, -1, kernel)

def highpass3(color, img):
    # make window
    cv.namedWindow('trackbar')
    
    # trackbar for thresholding
    cv.createTrackbar('th1', 'trackbar', 10, 255,
            dummy)
    cv.createTrackbar('th2', 'trackbar', 10, 255,
            dummy)

    # display in loop
    key = ord('a')
    cimg = img.copy()
    while key != ord('q'):
        th1 = cv.getTrackbarPos('th1', 'trackbar')
        th2 = cv.getTrackbarPos('th2', 'trackbar') 
        
        cimg = cv.Canny(img, th1, th2)
        kernel = np.ones((3, 3), np.uint8)
        cimg = cv.morphologyEx(cimg, cv.MORPH_CLOSE, kernel)
        cimg = cv.morphologyEx(cimg, cv.MORPH_DILATE, kernel)
        
        # HSV mask
        lower = np.array([0, 80, 0])
        upper = np.array([179, 255, 255])

        hsv = cv.cvtColor(color, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, lower, upper)
        #result = cv.bitwise_and(img, img, mask=mask)
        result = cv.add(mask, cimg)

        cv.imshow('trackbar', result)
        key = cv.waitKey(5)
 
    return cv.Canny(img, 20, 20)

color_img = cv.resize(cv.imread('train_img/img_007.jpg'), (0, 0), fx=0.3, fy=0.3)
gray_img = cv.cvtColor(color_img, cv.COLOR_BGR2GRAY)
img1 = highpass1(gray_img, 2)
img3 = highpass3(color_img, img1)


#cv.imshow('hpf1', img1)
#cv.imshow('hpf2', img2)
#cv.imshow('hpf3', img3)
cv.waitKey(0)

