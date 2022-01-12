import cv2 as cv
from matplotlib.pyplot import xcorr
import numpy as np
import queue as q
import data_backend_operations as db

class Node:
    def __init__(self, row, col, col_cls):
        self.row = row
        self.col = col
        self.cls = col_cls
    
    def __str__(self):
        return f"Row: {self.row} Column = {self.col} Color = {self.cls}" 


def get_segments(img_preproceessed):
    img_normalised = img_preproceessed / 255
    seg_description_val = 2
    for i in range(img_normalised.shape[0]):
        for j in (img_normalised.shape[1]):
            if img_normalised[i][j] == 1: 
                get_flood_fill_alg(img_normalised, i, j)
            else:
                pass


def get_flood_fill_alg(img_matrix, row, col, class_color, min_pxels=100, min_pix_flag=False):
    Q = q.Queue()
    Q.put(Node(row, col, 1))
    while not Q.empty():
        shifted_pix = []
        node = Q.get()
        if img_matrix[node.row][node.col] == 1:
            img_matrix[node.row][node.col] = class_color
            shifted_pix.append(node)
            new_node_list = [get_pixel_to_node(img_matrix, node.row - 1, node.col),\
                             get_pixel_to_node(img_matrix, node.row, node.col - 1),\
                             get_pixel_to_node(img_matrix, node.row, node.col + 1),\
                             get_pixel_to_node(img_matrix, node.row + 1, node.col)]
            for n_node in new_node_list:
                if n_node is not None:
                    Q.put(n_node)
        
        print("_______________")
    if min_pix_flag:
        if len(shifted_pix) > min_pxels:
            return img_matrix
        else:
            for node in shifted_pix:
                img_matrix[node.row][node.col] = 0
            return img_matrix
    else:
        return img_matrix


def get_pixel_to_node(img_matrix, row, col):
    if 0 <= row < img_matrix.shape[0] and 0 <= col < img_matrix.shape[1]:
        return Node(row, col, img_matrix[row][col])
    else:
        return None

