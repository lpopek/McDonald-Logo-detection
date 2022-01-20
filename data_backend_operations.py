from tkinter.tix import Tree
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import json


def get_img_from_dataset(no):
    img = cv.imread(f"dataset\image_{no}.jpg")
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

def print_img(img, title="domyslny"):
    cv.imshow(title, img)
    cv.waitKey()


def import_vector_for_cls():
    try:
        f = open('config.json')
        data = json.load(f)
        f.close()
        return data
    except FileNotFoundError:
        return None
