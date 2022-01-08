import cv2 as cv
from matplotlib import scale
import numpy as np
import matplotlib.pyplot as plt

from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage.measure import regionprops
import matplotlib.patches as mpatches
# Rectangular Kernel





rectangle_kernel = np.array([[1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1]], dtype=np.uint8)
# Elliptical Kernel
eliptical_kernel =np.array([[0, 0, 1, 0, 0],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [0, 0, 1, 0, 0]], dtype=np.uint8)
# Cross-shaped Kernel
cross_kernel = np.array([[0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0],
                        [1, 1, 1, 1, 1],
                        [0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0]], dtype=np.uint8)

cross_kernel_3x3= np.array([[0, 1, 0],
                            [1, 1 ,1],
                            [0, 1, 0]], dtype=np.uint8)


def get_img(no):
    img = cv.imread(f"dataset\image_{no}.jpg")
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    print("image loaded")
    return img

def resize_picture(img, scale_percent=20):
    print('Original Dimensions : ',img.shape)
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv.resize(img, dim, interpolation = cv.INTER_AREA)
    return resized



def converse_RGB2GRAY_SCALE(img):
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
    


def print_img(img, title="domyslny", gray_scale_flag = False):
    if(not gray_scale_flag):
        plt.title(title)
        plt.imshow(img)
        plt.show()
    else:
        plt.title(title)
        plt.imshow(img, cmap="gray")
        plt.show()

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

def make_binary_operations(img):
    dilation = cv.dilate(img, cross_kernel, iterations = 1)
    erosion = cv.erode(dilation, cross_kernel, iterations = 1)
    
    
    return erosion


def mark_regions(img, segments):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)
    i = 0 
    for region in regionprops(segments):
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', label="deer", linewidth=2)

        ax.add_patch(rect)
        i += 1
    plt.show()

def segmentation(img):
    segments_fz = felzenszwalb(img, scale=1000, sigma=0.5, min_size=500, multichannel=False)
    print(f"Felzenszwalb number of segments: {len(np.unique(segments_fz))}")
    plt.imshow(mark_boundaries(img, segments_fz))
    plt.title("Felzenszwalbs's method")
    plt.show()
    return segments_fz

def main():
    
    #print_img(img_0)
    for i in range(0, 15):
        base_img = get_img(i)
        img = cv.cvtColor(base_img, cv.COLOR_RGB2HSV)
        img = resize_picture(img)
        # img_gray = converse_RGB2GRAY_SCALE(img)
        # print_img(img_gray, "w skali szarości", gray_scale_flag=True)
        img_tresholded = get_treshold(img, 22, 8)
        img = make_binary_operations(img_tresholded)
        seg = segmentation(img_tresholded)
        mark_regions(img_tresholded, seg)
        


if __name__ == "__main__":
    main()