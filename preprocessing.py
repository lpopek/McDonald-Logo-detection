import cv2 as cv
import numpy as np

def get_kernel(size, rec_type="rectangle"):
    if rec_type == "rectangle":
        return np.ones((size, size), dtype=np.uint8)
    elif rec_type == "cross":
        kernel = np.zeros((size, size), dtype=np.uint8)
        mid = size // 2
        for row in kernel:
            row[mid] = 1
        for i in range(size):
            kernel[mid][i] = 1
        return kernel

print(get_kernel(7, rec_type="cross"))

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
    print("image conevrted from RGB to GRAY")
    return img_gray

def convert_RGB2HSV(img):
    r, g, b = cv.split(img)
    maxc = max(r, g, b)
    minc = min(r, g, b)
    v = maxc
    if minc == maxc:
        print("image conevrted from RGB to HSV")
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

    print("image conevrted from RGB to HSV")
    return h, s, v

def get_treshold(img, hue_center_value, hue_eps, saturation = 124):
    h, w, c= img.shape
    img_tresholded = np.zeros((h, w))
    i, j = 0, 0
    for row in img:
        for pix in row:
            if abs(pix[0] - hue_center_value) < hue_eps and pix[1] > saturation:
                img_tresholded[i][j] = 1.0
            j += 1
        j = 0
        i += 1
    print("treshold ready!")
    return img_tresholded

def erode_img(img, kernel):
    m, n = img.shape
    SE= np.ones((k,k), dtype=np.uint8)
    constant= (k-1)//2
    #Define new image
    imgErode= np.zeros((m,n), dtype=np.uint8)
    #Erosion without using inbuilt cv2 function for morphology
    for i in range(constant, m-constant):
        for j in range(constant,n-constant):
            temp= img[i-constant:i+constant+1, j-constant:j+constant+1]
            product= temp*SE
            imgErode[i,j]= np.min(product)