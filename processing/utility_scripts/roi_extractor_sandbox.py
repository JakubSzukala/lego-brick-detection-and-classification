import cv2 as cv 
import numpy as np
from matplotlib import pyplot as plt
from skimage import data, img_as_float
from skimage import exposure
import os


def dummy(value):
    pass

ath_kernel = 3 
def get_kernel(value):
    if not value % 2:
        value += 1

    global ath_kernel
    ath_kernel = value


def increase_contrast(img):
    try:
        gimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    except:
        gimg = img.copy()
    
    # Contrast stretch
    p2, p98 = np.percentile(gimg, (2, 98))
    img_rescale = exposure.rescale_intensity(gimg, in_range=(p2, p98)) # type: ignore
    #cv.imshow('contrast_stretch', img_rescale)
    
    # adaptive equalization
    img_adaptive_eq = exposure.equalize_adapthist(gimg, clip_limit=0.01)
    cv.imshow('adaptive_eq', img_adaptive_eq)
   
    # Gamma correction 
    img_gamma_corr = exposure.adjust_gamma(gimg, 3)
    cv.imshow('gamma', img_gamma_corr)

    # Logarithmic correction
    img_log = exposure.adjust_log(gimg, 5)
    cv.imshow('log', img_log)

    #cv.waitKey(0)
    return img_rescale


def adaptive_thresholding_experiment(img):
    # Validate
    #if img.shape[2] != 1:
    #    gimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #else:
    #    gimg = img.copy()
    gimg = img.copy()
    
    # Create a window with trackbars for kernel size and constant_c
    cv.namedWindow('ath_experiment')
    cv.createTrackbar('block_size', 'ath_experiment', 3, 21, get_kernel)
    cv.createTrackbar('constant_c', 'ath_experiment', 1000, 10000, dummy)
     
    # PROCESSING
    key = ord('a')
    while key != ord('q'):
        kernel = ath_kernel
        const_c = cv.getTrackbarPos('constant_c', 'ath_experiment') / 1000.0
        cimg = cv.bilateralFilter(gimg, 9, 150, 150)
        
        try:
            cimg = cv.adaptiveThreshold(cimg, 255, 
                    cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv.THRESH_BINARY_INV,
                    kernel, const_c)
        except:
            print('U cant do that')
        cimg = cv.medianBlur(cimg, 3)
        cimg = cv.blur(cimg, (5, 5))
        cimg = cv.Canny(cimg, 125, 255)
        
        # Improve lines
        #linesP = cv.HoughLinesP(cimg, 1, np.pi / 180, 30, None, 30, 15)
        #if linesP is not None:
        #    for i in range(0, len(linesP)):
        #        l = linesP[i][0]
        #        cv.line(cimg, (l[0], l[1]), (l[2], l[3]), (255,255,255), 3, cv.LINE_AA)
        """ 
        cimg = cv.erode(cimg, (3, 3),iterations=2)
        cimg = cv.medianBlur(cimg, 3)
        cimg = cv.morphologyEx(cimg, cv.MORPH_OPEN, (5, 5), iterations=1)
        cimg = cv.Canny(cimg, 255, 255.0 / 3.0)
        cimg = cv.dilate(cimg, (5, 5), iterations=2)
        
        h, w = cimg.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)

        cv.floodFill(cimg, mask, (0,0), 255);
        """
    # CONTOUR DETECTION  
        cv.imshow('ath_experiment', cimg)
        key = cv.waitKey(1)
    
    contours, _ = cv.findContours(cimg, 
            cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    # Get centroids and mark them with text
    centroids = np.empty((0, 2), dtype=np.float32) # conflict with int()?
    for index, c in enumerate(contours):
        cv.drawContours(img, c, -1, (0, 150, 0), thickness=3)
    print(len(contours))
    cv.imshow('Centroids', img)
    cv.waitKey(0)
    

def watershed(img):
    gimg = img.copy()
    #cimg = cv.bilateralFilter(gimg, 9, 150, 150)
    cimg = cv.adaptiveThreshold(gimg, 255, 
                    cv.ADAPTIVE_THRESH_MEAN_C, 
                    cv.THRESH_BINARY_INV,
                    17, 4.5)
    cimg = cv.medianBlur(cimg, 3)
    
    kernel = np.ones((3, 3), np.uint8)
    cimg = cv.morphologyEx(cimg, cv.MORPH_OPEN, kernel, iterations=2)
    #cimg = cv.dilate(cimg, (3, 3), iterations=2) 
    
    #cv.imshow('watershed_try', cimg)

    #cimg = cv.boxFilter(cimg, -1, (15, 15), normalize=False)
    cimg = cv.bitwise_not(cimg)
    cv.imshow('inverted', cimg)

    fg_estimation = cv.distanceTransform(cimg, cv.DIST_L2, 5)
    cv.normalize(fg_estimation, fg_estimation, 0, 1.0, cv.NORM_MINMAX)
    
    #fg_estimation = increase_contrast(fg_estimation)
    
    fg_estimation = (fg_estimation * 255).astype(np.uint8) 
    fg_estimation = cv.equalizeHist(fg_estimation) 
    _, th = cv.threshold(fg_estimation, 90, 255, cv.THRESH_BINARY)

    th = cv.dilate(th, (5, 5), iterations=50) 
    #th = cv.boxFilter(th, -1, (15, 15), normalize=False)
    #th = cv.boxFilter(th, -1, (15, 15), normalize=False)
    #th = cv.boxFilter(th, -1, (15, 15), normalize=False)
    #th = cv.boxFilter(th, -1, (15, 15), normalize=False)
    #th = cv.erode(th, (5, 5), iterations=20)



    cv.imshow('dist_transform', th)
    cv.waitKey(0)


def polyline_contour_closing(img):
    gimg = img.copy()
    cimg = cv.adaptiveThreshold(gimg, 255, 
                    cv.ADAPTIVE_THRESH_MEAN_C, 
                    cv.THRESH_BINARY_INV,
                    17, 4.5)
    kernel = np.ones((3, 3), np.uint8)
    cimg = cv.morphologyEx(cimg, cv.MORPH_OPEN, kernel, iterations=2)
    img = cv.medianBlur(cimg, 3)
    
    cv.imshow('initial_th', cimg)
    cv.waitKey(0)

    cimg = cv.Canny(cimg, 125, 255)
    cv.imshow('canny', cimg)
    cv.waitKey(0)

def harris_corner(img):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    
    gray = np.float32(gray)
    dst = cv.cornerHarris(gray, 2, 3, 0.04)
    dst = cv.dilate(dst, None)

    ret, dst = cv.threshold(dst,0.01*dst.max(),255,0)
    dst = np.uint8(dst)
    # find centroids
    ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst)
    # define the criteria to stop and refine the corners
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
    # Now draw them
    res = np.hstack((centroids,corners))
    res = np.int0(res)
    img[res[:,1],res[:,0]]=[0,0,255]
    img[res[:,3],res[:,2]] = [0,255,0]
    
    cv.imshow('corners', img)
    cv.waitKey()

def shitomasi(img):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    p2, p98 = np.percentile(gray, (2, 98))
    gray = exposure.rescale_intensity(gray, in_range=(p2, p98)) # type: ignore
    corners = cv.goodFeaturesToTrack(gray,600,0.01,10)
    corners = np.int0(corners)
    for i in corners:
        x,y = i.ravel()
        cv.circle(img,(x,y),3,255,-1)
    plt.imshow(img),plt.show()


def sobel(img):
    # TODO: Beautify this 
    color_img = img.copy()
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    p2, p98 = np.percentile(img, (2, 98))
    img = exposure.rescale_intensity(img, in_range=(p2, p98)) # type: ignore
    
    M_x = np.array([
        [1.0, 0.0, -1.0],
        [2.0, 0.0, -2.0],
        [1.0, 0.0, -1.0]
        ], dtype=np.float32)

    M_y = np.array([
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, -2.0, -1.0]
        ], dtype=np.float32)
    
    M_45deg = np.array([
        [0.0, 1.0, 2.0],
        [-1.0, 0.0, 1.0],
        [-2.0, -1.0, 0.0]
        ])

    divisor = 4.0

    copy = img.copy()
    copy = copy.astype('float32')
    
    ddepth = cv.CV_32F

    out = [
            abs(cv.filter2D(copy, ddepth, M_x) / divisor),
            abs(cv.filter2D(copy, ddepth, M_y) / divisor),
            abs(cv.filter2D(copy, ddepth, M_45deg / divisor)) # Unnecessary?
            ]
    # TODO: change that... it is confusing 
    out.append(cv.add(out[0], out[1], out[2])) 
    
    # Bring back to 8bit for displaying
    out[0] = out[0].astype('uint8')
    out[1] = out[1].astype('uint8')
    out[2] = out[2].astype('uint8')
    out[3] = out[3].astype('uint8')
    
    # Adjust contrast little bit more 
    p2, p98 = np.percentile(out[3], (2, 98))
    output = exposure.rescale_intensity(out[3], in_range=(p2, p98)) # type: ignore
    cv.imshow('sobel', output)
    cv.waitKey()

    # NICE
    _, output = cv.threshold(output, 130, 255, cv.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    output = cv.morphologyEx(output, cv.MORPH_CLOSE, kernel, iterations=1)  
    output = cv.morphologyEx(output, cv.MORPH_DILATE, kernel, iterations=1)  
    
    contours, _ = cv.findContours(output, 
        cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    counter = 0
    for index, c in enumerate(contours):
        if len(c) > 30:
            # Minimal area rectangle 
            #rect = cv.minAreaRect(c)
            #box = cv.boxPoints(rect)
            #box = np.int0(box)
            #cv.drawContours(color_img, [box], -1, (0, 0, 150), thickness=3)

            # Area rectangle 
            x, y, w, h = cv.boundingRect(c)
            cv.rectangle(color_img, (x, y), (x + w, y + h), (0, 0, 150), thickness=3)

            # Contours 
            cv.drawContours(color_img, [c], -1, (0, 150, 0), thickness=3)
            counter += 1

    print(counter)

    # Display
    cv.imshow('contours', color_img)
    cv.imshow('prewitt', output)
    cv.waitKey()
    

def background_subtraction():
    images_dir = 'train_img'
    dir_content = os.listdir(images_dir) 
    dir_content.sort()
    
    back_sub = cv.createBackgroundSubtractorMOG2()
     
    for img in dir_content:
        img_path = os.path.join(images_dir, img)
        img = cv.resize(cv.imread(img_path), (0, 0), fx=0.3, fy=0.3)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = cv.bilateralFilter(img, 9, 150, 150)
        fg_mask = back_sub.apply(img)

        cv.imshow('image', img)
        cv.imshow('mask', fg_mask)

        cv.waitKey(0)


if __name__ == '__main__':
    img = cv.resize(cv.imread('train_img/img_007.jpg'), (0, 0), fx=0.3, fy=0.3)
    #adjusted_cont = increase_contrast(img.copy())
    #sobel(img)
    initial = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 
    adjusted_cont = increase_contrast(initial)
    cv.imshow('contrast', adjusted_cont)
    cv.imshow('initial', initial)
    cv.waitKey()
    #background_subtraction()
