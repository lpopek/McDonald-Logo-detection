import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import json
from random import randint


def get_img_from_dataset(no):
    img = cv.imread(f"dataset\image_{no}.jpg")
    return img

def resize_picture(img, scale_percent=20):
    print('original Dimensions:', img.shape)
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    #TODO make interpolation using opencv
    resized = cv.resize(img, dim, interpolation = cv.INTER_AREA)
    return resized

def print_img(img, title="domyslny", gray_scale_flag=False, BGR=True):
    if not gray_scale_flag:
        if BGR is True:
            plt.title(title)
            plt.imshow(img)
            plt.show()
    else:
        plt.title(title)
        plt.imshow(img, cmap="gray")
        plt.show()

def create_bbox(img, p_min, p_max, thickness=2, color=(255, 0, 0)):
    for i in range(p_min[0], p_max[0]):
        for j in range(p_min[1], p_min[1] + thickness):
            img[j][i] = color
    for i in range(p_min[0], p_max[0]):
        for j in range(p_max[1] - thickness, p_max[1]):
            img[j][i] = color
    for i in range(p_min[1], p_max[1]):
        for j in range(p_min[0], p_min[0] + thickness):
            img[i][j] = color
    for i in range(p_min[1], p_max[1]):
        for j in range(p_max[0] - thickness, p_max[0]):
            img[i][j] = color
    return img

def import_vector_for_cls():
    try:
        f = open('config.json')
        data = json.load(f)
        f.close()
        return data
    except FileNotFoundError:
        return None

