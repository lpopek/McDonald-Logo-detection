import numpy as np
from Node import Node



def get_area(bbox_image):
    area = 0
    for row in bbox_image:
        for pix in row:
            if pix:
                area += 1
    return area