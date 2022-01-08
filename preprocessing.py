import cv2 as cv
import numpy as np

def invert_opencv_split(img):
    b, g, r = cv.split(img)
    return cv.merge(r, g, b)

def convert_RGB2GRAY(img):
    h, w, c= img.shape
    img_gray = np.zeros((h, w))
    i, j  = 0, 0
    for row in img:
        for pix in row:
            img_gray[i][j] = sum(pix)/3
            j += 1
        i += 1
        j = 0
    return img_gray

def convert_RGB2HSV(img):
    r, g, b = cv.split(img)
    maxc = max(r, g, b)
    minc = min(r, g, b)
    v = maxc
    if minc == maxc:
        return 0.0, 0.0, v
    s = (maxc-minc) / maxc
    rc = (maxc-r) / (maxc-minc)
    gc = (maxc-g) / (maxc-minc)
    bc = (maxc-b) / (maxc-minc)
    if r == maxc:
        h = bc-gc
    elif g == maxc:
        h = 2.0 + rc - bc
    else:
        h = 4.0 + gc - rc
    h = (h / 6.0)  % 1.0
    return h, s, v