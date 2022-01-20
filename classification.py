import numpy as np
from Node import Node

def get_perimiter(bbox_image):
    perimiter = 0
    for row in bbox_image:
        for pix in row:
            if pix == 1:
                perimiter += 1
    return perimiter

def get_area(bbox_image):
    area = 0
    for row in bbox_image:
        for pix in row:
            if pix:
                area += 1
    return area