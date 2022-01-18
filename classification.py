import numpy as np

dict_trans = {
    0:0,
    1:1,
    2:2,
    3:5,
    4:8,
    5:7,
    6:6,
    7:3
}

def get_matrix_around_pixel(bbox_image, point):
    row, col = point[0], point[1]
    if 0 < row < bbox_image.shape[0] - 1 and 0 < col < bbox_image.shape[1] - 1:
        M = bbox_image[row - 1: row + 2, col - 1: col + 2]
        return bbox_image[row - 1: row + 2, col - 1: col + 2]
    else:
        return None

def get_clockwise_list_from_matrix(M, rot):
    cl_list = np.concatenate((np.append(M[0], M[1][2]), np.append(M[2][::-1], np.array(M[1][0]))))
    return np.roll(cl_list, -rot)

def augment_black_border(bbox_image):
    bordered_img = np.zeros((bbox_image.shape[0] + 2, bbox_image.shape[1] + 2))
    bordered_img[1:-1, 1:-1] = bbox_image
    return bordered_img

def get_rotation():
    pass
        
def get_cord_from_M_position(i, j, M):
    pass




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
                s = (i, j)
                border.append(s)
                p = s
                perimeter += 1
                break
            j += 1
        else:
            i += 1
            continue
        break
    if p is not None:
        M = get_matrix_around_pixel(bbox_image, p)
        c_val = M[1][0]
        c = (i, j - 1)
        while s != c:
            M_list  = get_clockwise_list_from_matrix(M, 0)
            for m in M_list:
                if m == ann:
                    border.append(c)
                    perimeter += 1
                    break
                prev_step = m


    

    # return coutour_img, perimeter

def get_perimeter(bbox_image, annotation):
    pass


def get_area(bbox_image):
    area = 0
    for row in bbox_image:
        for pix in row:
            if pix:
                area += 1
    return area