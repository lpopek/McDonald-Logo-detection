from cgi import print_directory
import numpy as np
import cv2 as cv
from Node import Node

def calculate_perimiter(bbox_image):
    perimiter = 0
    for row in bbox_image:
        for pix in row:
            if pix == 1:
                perimiter += 1
    return perimiter

def calculate_area(segment):
    return len(segment["cordinates"])

def calculate_Malinowska_ratio(area, perimeter):
    return perimeter/(2 * np.sqrt(np.pi * area)) - 1

def calculate_moment(p, q, segment):
    m = 0
    for node in segment["cordinates"]:
        m += node.row**p * node.col**q
    return m

def find_center(segment):
    m10 = calculate_moment(1, 0, segment)
    m01 = calculate_moment(0, 1, segment)
    m00 = calculate_moment(0, 0, segment)
    return m10/m00, m01/m00

def calculate_central_moment(segment, p, q, row_center=-1, col_center=-1):
    m_center = 0
    if row_center == -1 or col_center == -1:
        row_center, col_center = find_center(segment)
    for node in segment["cordinates"]:
        m_center += (node.row - row_center)**p * (node.col - col_center)**q
    return m_center

def calculate_invariants(segment):
    results = [0] * 11
    
    m00 = calculate_moment(0, 0, segment)
    row_center, col_center = find_center(segment)
    M = np.zeros((4, 4))
    for p in range(4):
        for q in range(4):
            M[p, q] = calculate_central_moment(segment, p, q, row_center, col_center)
    
    #M1
    results[1] = (M[2, 0] + M[0, 2]) / m00**2
    #M2
    results[2] = ((M[2, 0] + M[0, 2])**2 + 4 *  M[1, 1] ** 2) / m00**4
    #M3
    results[3] = ((M[3, 0] - 3 * M[1, 2])**2 + (3 * M[2, 1] - M[0, 3])**2) / m00**5
    #M4
    results[4] = ((M[3, 0] + M[1, 2])**2 + (M[2, 1] + M[0, 3])**2) / m00**5    
    #M5
    results[5] = ((M[3, 0]- 3 * M[1, 2]) * (M[3, 0] + M[1, 2]) * 
                  ((M[3, 0]+ M[1, 2])**2 - 3 * (M[2, 1] + M[0, 3])**2) +
                 (3 * M[2, 1] - M[0, 3]) * (M[2, 1] + M[0, 3]) * 
                  (3 * (M[3, 0] + M[1, 2])**2 - (M[2, 1] + M[0, 3])**2)
                 ) / m00** 10
    #M6
    results[6] = ((M[2, 0] - M[0, 2])*((M[3, 0] + M[1, 2])**2 - (M[2, 1] + M[0, 3])**2) +
                 4 * M[1, 1] * (M[3, 0] + M[1, 2]) * (M[2, 1] + M[0, 3])) / m00**7
    #M7
    results[7] = (M[2, 0] * M[0, 2] - M[1, 1]**2) / m00**4
    #M8
    results[8] = (M[3, 0] * M[1, 2] +  M[2, 1] * M[0, 3] - M[1, 2]**2 - M[2, 1]**2) / pow(m00, 5)    
    #M9
    results[9] = (M[2, 0] * (M[2, 1] * M[0, 3] - M[1, 2]**2) + 
                 M[0, 2] * (M[0, 3] * M[1, 2] - M[2, 1]**2) -
                 M[1, 1] * (M[3, 0] * M[0, 3] - M[2, 1] * M[1, 2])) / m00**7
    #M10
    results[10] = ((M[3, 0] * M[0, 3] - M[1, 2] * M[2, 1])**2 - 
                  4*(M[3, 0]*M[1, 2] - M[2, 1]**2)*(M[0, 3] * M[2, 1] - M[1, 2])) / m00**10
    
    print("Vector of invariants values calculated")
    return results

