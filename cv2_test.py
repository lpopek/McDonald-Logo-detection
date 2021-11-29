import cv2 as cv
from matplotlib import scale
import numpy as np
import matplotlib.pyplot as plt

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

def plot_few_images(img_list):
    w = 10
    h = 10
    fig = plt.figure(figsize=(8, 8))
    columns = 4
    rows = 5
    for i in range(1, columns*rows +1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
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
        fig = plt.figure(figsize=(2, 1))
        fig.add_subplot(2, 1, 1)
        plt.imshow(img_tresholded)
        fig.add_subplot(2, 1, 2)
        plt.imshow(img)
        plt.show()
        


if __name__ == "__main__":
    main()