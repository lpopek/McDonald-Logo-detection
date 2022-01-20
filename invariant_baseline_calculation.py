import cv2 as cv
import numpy as np
import os
import json
import preprocessing as pr
import segmentation as seg
import classification as cls
import data_backend_operations as db

def get_invariants_for_McDonalds_logo(dataset_path=None):
    if dataset_path is None:
        img_name_list = os.listdir("train\\train_dataset")
        img_list = [cv.imread(f"train\\train_dataset\\{img_el}")for img_el in img_name_list]
            
    else:
        img_name_list = os.listdir(dataset_path)
        img_list = [cv.imread(f"{dataset_path}\\{img_el}")for img_el in img_name_list]
    invariant_matrix = []
    for img in img_list:
        img_gray = pr.convert_BGR2GRAY(img)
        img_segmented, seg_no, segments = seg.get_segments(img_gray)
        for segment in segments:
            segm_bbox = seg.crop_segment(img_segmented, segment["cordinates"])
            segm_bbox_border, perimeter, _ = seg.get_Moore_Neighborhood_countour(segm_bbox, segment["key"])
            db.print_img(segm_bbox_border)
            area = cls.calculate_area(segment)
            invariants = cls.calculate_invariants(segment)
            invariants[0] = cls.calculate_Malinowska_ratio(area, perimeter)
        invariant_matrix.append(invariants)
    return np.array(invariant_matrix)

def calculate_vector_with_sigm(inv_mat):
    vect, feature = inv_mat.shape
    median_vector = np.zeros(feature)
    sigma_vector = np.zeros(feature)
    for j in range(feature):
        helper = np.zeros(vect)
        for i in range(vect):
            median_vector[j] += inv_mat[i][j]/ vect
            helper[i] = inv_mat[i][j]
        sigma_vector[j] = np.std(helper)
    
    return median_vector, sigma_vector

def get_config_cls_file(median_vector, sigma_vector):
    data = {"feature value": median_vector.tolist(), "standard deviation": sigma_vector.tolist()}
    with open("config.json", "w") as write_file:
        json.dump(data, write_file, indent=4)
    return True

if __name__ == "__main__":
    inv = get_invariants_for_McDonalds_logo()
    v1, v2 = calculate_vector_with_sigm(inv)
    get_config_cls_file(v1, v2)
