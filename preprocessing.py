import cv2 as cv
import numpy as np

def get_extremal_channel_value(channel_1, channel_2, is_max = True):
    if is_max is True:
        max_c1 = np.amax(np.array(channel_1))
        max_c2 = np.amax(np.array(channel_2))
        return max_c1 if max_c2 < max_c1 else max_c2
    else:
        min_c1 = np.amin(np.array(channel_1))
        min_c2 = np.amin(np.array(channel_2))
        return min_c2 if min_c2 < min_c1 else min_c1


def get_kernel(size, kern_type="rectangle"):
    if kern_type == "rectangle":
        return np.ones((size, size), dtype=np.uint8)
    elif kern_type == "cross":
        kernel = np.zeros((size, size), dtype=np.uint8)
        mid = size // 2
        for row in kernel:
            row[mid] = 1
        for i in range(size):
            kernel[mid][i] = 1
        return kernel


def invert_opencv_split(img):
    b, g, r = cv.split(img)
    return cv.merge(r, g, b)

def convert_BGR2GRAY(img):
    h, w, c= img.shape
    img_gray = np.zeros((h, w), dtype=np.uint8)
    i, j  = 0, 0
    for row in img:
        for pix in row:
            img_gray[i][j] = pix[0] * 0.114 + pix[1] * 0.587 + pix[2] * 0.299
            j += 1
        i += 1
        j = 0
    print("image converted from BGR to GRAY")
    return img_gray

def convert_BGR2HSV(img, is_RGB = False):
    hsv_img = np.zeros(img.shape, dtype=np.uint8)
    i, j = 0, 0
    for row in img:
        for pix in row:
            if is_RGB is True:
                r, g, b = pix[2]/255.0, pix[1]/255.0, pix[0]/255.0
            else:
                b, g, r = pix[0]/255.0, pix[1]/255.0, pix[2]/255.0
            cmax = max(r, g, b)   
            cmin = min(r, g, b)    
            diff = cmax - cmin       
            if cmax == cmin:
                h = 0
            elif cmax == r:
                h = ((60 * (g - b) / diff)) 
            elif cmax == g:
                h = ((60 * (b - r) / diff) + 120) 
            elif cmax == b:
                h = ((60 * (r - g) / diff) + 240) 
            if h < 0 : h += 360
            if cmax == 0:
                s = 0
            else:
                s = (diff / cmax)
            v = cmax
            hsv_img[i][j] =  h // 2, 255 * s, 255 *v
            j += 1
        i += 1
        j = 0
    print("image converted from BGR to HSV")
    return hsv_img

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

def erode_img(img, kern_size=7,  kern_type="rectangle"):
    if kern_type == "rectangle":
        kernel = get_kernel(kern_size)
    elif kern_type == "cross":
        kernel = get_kernel(kern_size, kern_type=kern_type)
    k_const = (kern_size - 1) // 2
    img_erode = np.zeros(img.shape, dtype=np.uint8)
    for i in range(k_const, img.shape[0] - k_const):
        for j in range(k_const, img.shape[1] - k_const):
            temp = img[i-k_const:i+k_const+1, j-k_const:j+k_const+1]
            product= temp * kernel
            img_erode[i,j]= np.min(product)
    return img_erode

def dilate_img(img, kern_size=7,  kern_type="rectangle"):
    img_dilate= np.zeros(img.shape, dtype=np.uint8)
    if kern_type == "rectangle":
        kernel = get_kernel(kern_size)
    elif kern_type == "cross":
        kernel = get_kernel(kern_size, kern_type=kern_type)
    k_const = (kern_size - 1) // 2
    for i in range(k_const, img.shape[0]- k_const):
        for j in range(k_const, img.shape[1] - k_const):
            temp = img[i-k_const:i+k_const+1, j-k_const:j+k_const+1]
            product = temp * kernel
            img_dilate[i,j] = np.max(product)
    return img_dilate


def make_binary_operations(img):
    dilated_image = dilate_img(img)
    #dilated_image_cros = dilate_img(dilated_image, kern_type= "cross")
    eroded_img = erode_img(dilated_image)
    print("closing operation")
    return eroded_img

def Sobel_filter(img, is_BGR = False):
    kernel_X = np.array([-1, 0, 1],\
                        [-2, 0, 2],\
                        [-1, 0, 1])


    kernel_Y = np.array([1,  2,  1],\
                        [0,  0,  0],\
                        [-1, -2, -1])
    if is_BGR is not True:
        for i in range(1, img.shape[0] - 1):
            for j in range(1, img.shape[1] - 1):
                temp = img[i-1:i+2, j-1:j+2]
                x_prod = temp * kernel_X
                y_prod = temp * kernel_Y
                new_value = np.sqrt(x_prod**2 + y_prod **2)