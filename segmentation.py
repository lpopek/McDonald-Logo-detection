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

    def __eq__(self, other):
        return ((self.row, self.col, self.cls) == (other.row, other.col, other.cls))


def get_segments(img_preproceessed, min_pix=100, min_pixel_flag = True):
    img_normalised = img_preproceessed / np.amax(img_preproceessed)
    segment_list = []
    seg_description_val = 2
    for row in range(img_normalised.shape[0]):
        for col in range(img_normalised.shape[1]):
            if img_normalised[row][col] == 1: 
                img_normalised, is_new_segm_added, new_seg = get_flood_fill_alg(img_normalised,\
                     row, col, seg_description_val,\
                     min_pixels=min_pix, min_pix_flag=min_pixel_flag)
                if is_new_segm_added is True:
                    seg_description_val += 1
                    segment_list.append(new_seg)
            else:
                pass
    return img_normalised, seg_description_val - 2, segment_list


def get_flood_fill_alg(img_matrix, row, col, class_color, min_pixels=100, min_pix_flag=False):
    Q = q.Queue()
    Q.put(Node(row, col, 1))
    shifted_pix = []
    while not Q.empty():
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
    if min_pix_flag:
        if len(shifted_pix) > min_pixels:
            return img_matrix, True, shifted_pix
        else:
            for node in shifted_pix:
                img_matrix[node.row][node.col] = 0
            return img_matrix, False, None
    else:
        return img_matrix, True, []


def get_pixel_to_node(img_matrix, row, col):
    if 0 <= row < img_matrix.shape[0] and 0 <= col < img_matrix.shape[1]:
        return Node(row, col, img_matrix[row][col])
    else:
        return None

def take_row_node(node):
    return node.row

def take_col_node(node):
    return node.col

def determine_extreme_points_seg(segment):
    seg_sort_row = sorted(segment, key=take_row_node)
    seg_sort_col = sorted(segment, key=take_col_node)

    return (seg_sort_col[0].col, seg_sort_row[0].row),\
          (seg_sort_col[-1].col, seg_sort_row[-1].row)

