import cv2 as cv
from matplotlib.pyplot import xcorr
import numpy as np
import queue as q
import data_backend_operations as db
from Node import Node


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
                    segment_list.append({"cordinates": new_seg, "key": seg_description_val})
                    seg_description_val += 1
            else:
                pass
    print("segmentation finished")
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

def crop_segment(img, segment):
    p1, p2 = determine_extreme_points_seg(segment)
    return img[p1[1]:p2[1], p1[0]:p2[0]]

def get_matrix_around_pixel(bbox_image, point):
    row, col = point[0], point[1]
    if 0 < row < bbox_image.shape[0] - 1 and 0 < col < bbox_image.shape[1] - 1:
        return bbox_image[row - 1: row + 2, col - 1: col + 2]
    else:
        return None

def get_Node_matrix_around_pixel(bbox_image, row, col):
    M = []
    k = 0
    if 0 < row < bbox_image.shape[0] - 1 and 0 < col < bbox_image.shape[1] - 1:
        for i in range(row - 1, row + 2):
            M.append([])
            for j in range(col - 1, col + 2):
                M[k].append(Node(i, j, bbox_image[i][j]))
            k += 1
        return np.array(M)
    else:
        return None

def get_clockwise_list_from_matrix(M, rot):
    cl_list = np.concatenate((np.append(M[0], M[1][2]), np.append(M[2][::-1], np.array(M[1][0]))))
    return np.roll(cl_list, -rot)

def augment_black_border(bbox_image):
    bordered_img = np.zeros((bbox_image.shape[0] + 2, bbox_image.shape[1] + 2))
    bordered_img[1:-1, 1:-1] = bbox_image
    return bordered_img

def get_rotation_from_direction(c, p):
    """
    Get rotation value for M neighbourhood list from direction of vector pointing from p -> c
    """
    vec = [c.row - p.row, c.col - p.col]
    if vec[0] > 0 and vec[1] == 0:
        return 5
    elif vec[0] < 0 and vec[1] == 0:
        return 1
    elif vec[0] == 0 and vec[1] > 0:
        return 3
    elif vec[0] == 0 and vec[1] < 0:
        return 7
    else:
        raise ValueError()

def search_neigbourhood(p, c, s, bbox_image, ann):
    M = get_Node_matrix_around_pixel(bbox_image, p.row, p.col)
    rot = get_rotation_from_direction(c, p)
    M_list = get_clockwise_list_from_matrix(M, rot)
    for m in M_list:
        if m == s:
            return c, p, False
        if m.cls == ann:
            p_new = m
            break
        c_new = m
    return c_new, p_new, True

def get_Moore_Neighborhood_countour(bbox_image, ann):
    bbox_image = augment_black_border(bbox_image)
    p = None
    perimeter = 0
    border = []
    i, j = 0, 0
    for row in bbox_image:
        for pix in row:
            if pix == ann:
                #bbox_image[i][j] = 1
                s = Node(i, j, ann)
                border.append(s)
                p = s
                break
            j += 1
        else:
            i += 1
            j = 0 
            continue
        break
    if p is not None:
        c = Node(i, j - 1, bbox_image[i][j - 1])
        continue_flag = True
        while continue_flag:
            c, p, continue_flag = search_neigbourhood(p, c, s, bbox_image, ann)
            #bbox_image[p.row][p.col] = 1
            border.append(p)
            perimeter += 1
        for node in border:
            bbox_image[node.row][node.col] = 1
    return bbox_image[1:-1, 1:-1], perimeter, border
