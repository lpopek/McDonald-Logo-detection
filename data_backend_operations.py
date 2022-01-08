import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def get_image(image_path):
    img = cv.imread(image_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    print("image loaded")
    return img

def get_img_from_dataset(no):
    img = cv.imread(f"dataset\image_{no}.jpg")
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    print("image loaded")
    return img

def resize_picture(img, scale_percent=20):
    print('Original Dimensions : ',img.shape)
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    #TODO make interpolation using opencv
    resized = cv.resize(img, dim, interpolation = cv.INTER_AREA)
    return resized

def print_img(img, title="domyslny", gray_scale_flag = False):
    if(not gray_scale_flag):
        plt.title(title)
        plt.imshow(img)
        plt.show()
    else:
        plt.title(title)
        plt.imshow(img, cmap="gray")
        plt.show()
