import numpy as np
from Node import Node

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
                bbox_image[i][j] = 1
                s = Node(i, j, 1)
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
            bbox_image[p.row][p.col] = 1
            border.append(p)
            perimeter += 1
    return bbox_image[1:-1, 1:-1], perimeter, border

def get_area(bbox_image):
    area = 0
    for row in bbox_image:
        for pix in row:
            if pix:
                area += 1
    return area