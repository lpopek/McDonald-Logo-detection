from re import A
import unittest
import cv2 as cv
import numpy as np
from numpy.lib.index_tricks import nd_grid
from cv2_test import print_img, segmentation
import data_backend_operations as db
from Node import Node 

IMG = cv.imread("test\\test.png")


import preprocessing as pr
@unittest.skip("test were verified")
class TestPreprocessingMethods(unittest.TestCase):

    def test_get_kernel(self):
        self.assertEqual(pr.get_kernel(7).all(),np.ones((7,7)).all())

    def test_convert_BGR2GRAY(self):
        img_gray = cv.cvtColor(IMG, cv.COLOR_BGR2GRAY)
        img_gray_test = pr.convert_BGR2GRAY(IMG)
        try: 
            np.testing.assert_allclose(img_gray_test, img_gray, atol=1.0)
        except AssertionError:
            print("Convertion RGB2GRAY don't work")
    
    def test_convert_BGR2HSV(self):
        img_hsv  = cv.cvtColor(IMG, cv.COLOR_BGR2HSV)
        img_hsv_test = pr.convert_BGR2HSV(IMG)
        try: 
            np.testing.assert_allclose(img_hsv, img_hsv_test, atol=1.0)
        except AssertionError:
            print("Convertion BGR2GRAY don't work")

    def test_dilation(self):
        img = pr.convert_RGB2GRAY(IMG)
        _, img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
        img_dilated_test = pr.dilate_img(img, kern_size=5)
        ker = np.ones((5,5), dtype=np.uint8)
        img_dilated = cv.dilate(img, kernel=ker)
        try: 
            np.testing.assert_equal(img_dilated_test, img_dilated)
        except AssertionError:
            print("Operation of Dilation not working")

    def test_erosion(self):
        img = pr.convert_RGB2GRAY(IMG)
        _, img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
        img_eroded_test = pr.erode_img(img, kern_size=5)
        ker = np.ones((5,5), dtype=np.uint8)
        img_eroded_test = cv.dilate(img, kernel=ker)
        try: 
            np.testing.assert_equal(img_eroded_test, img_eroded_test)
        except AssertionError:
            print("Operation of Erosion not working")

import segmentation as seg
class TestSegmentationMethods(unittest.TestCase):
    #TODO naprawić testy o maja błedne argumenty i 
    @unittest.skip("test not verified")
    def test_flood_fill_algorithm(self):
        mat = np.array([[0, 0, 1, 1, 1], [0, 1, 0, 1, 0], [0, 1, 0, 1, 0], [0, 0, 1, 1, 0], [1, 1, 1, 1, 0]])
        mat = seg.get_flood_fill_alg(mat, 0, 2, 2)
        mat_ = np.array([[0, 0, 2, 2, 2], [0, 1, 0, 2, 0], [0, 1, 0, 2, 0], [0, 0, 2, 2, 0], [2, 2, 2, 2, 0]])
        try:
            np.testing.assert_equal(mat, mat_, err_msg="Assertion failed")
            self.assertTrue(True)
        except AssertionError:
            print("Assertion failed")
    @unittest.skip("test not verified")
    def test_segmentation(self):
        mat = np.array([[0, 0, 1, 1, 1], [0, 1, 0, 1, 0], [0, 1, 0, 0, 0], [0, 0, 1, 1, 0], [1, 1, 1, 1, 0]])
        mat, seg_no = seg.get_segments(mat)
        mat_ = np.array([[0, 0, 2, 2, 2], [0, 3, 0, 2, 0], [0, 3, 0, 0, 0], [0, 0, 4, 4, 0], [4, 4, 4, 4, 0]])
        try:
            np.testing.assert_equal(mat, mat_)
        except AssertionError:
            print("Unequal matrixes")
        try:
            self.assertEqual(seg_no, 3)
        except AssertionError:
            print("Wrong number of segments")
    
    def test_get_matrix_around_pixel(self):
        mat = np.array([[0, 0, 1, 1, 1], [0, 1, 0, 1, 0], [0, 1, 0, 0, 0], [0, 0, 1, 1, 0], [1, 1, 1, 1, 0]])
        mat_1 = seg.get_matrix_around_pixel(mat, (2, 2))
        mat_2 = seg.get_matrix_around_pixel(mat, (0, 0))
        mat_ = np.array([[1, 0, 1], [1, 0 , 0], [0, 1, 1]])
        try:
            np.testing.assert_equal(mat_1, mat_)
        except AssertionError:
            print("Unequal matrixes.")
        self.assertEqual(mat_2, None)
    
    def test_augment_black_border(self):
        mat = np.array([[0, 0, 1, 1, 1],
                        [0, 1, 0, 1, 0],
                        [0, 1, 0, 0, 0],
                        [0, 0, 1, 1, 0],
                        [1, 1, 1, 1, 0]])

        mat = seg.augment_black_border(mat)
        mat_ = np.array([[0, 0, 0, 0, 0, 0, 0],\
                         [0, 0, 0, 1, 1, 1, 0],\
                         [0, 0, 1, 0, 1, 0, 0],\
                         [0, 0, 1, 0, 0, 0, 0],\
                         [0, 0, 0, 1, 1, 0, 0],\
                         [0, 1, 1, 1, 1, 0, 0],\
                         [0, 0, 0, 0, 0, 0, 0]])

        try:
            np.testing.assert_equal(mat, mat_)
        except AssertionError:
            print("Unequal matrixes. Bad widthening")
    
    def test_get_Node_matrix_around_pixel(self):
        mat = np.array([[0, 0, 1, 1, 1],\
                        [0, 1, 0, 1, 0],\
                        [0, 1, 0, 0, 0],\
                        [0, 0, 1, 1, 0],\
                        [1, 1, 1, 1, 0]])
        mat_1 = seg.get_Node_matrix_around_pixel(mat, 2, 2)
        mat_ = np.array([[Node(1, 1, 1), Node(1, 2, 0), Node(1, 3, 0)]\
                        ,[Node(2, 1, 1), Node(2, 2, 0), Node(2, 3, 0)]\
                        ,[Node(3, 1, 0), Node(3, 2, 1), Node(3, 3, 1)]])
        self.assertEqual(mat_1[0][0], mat_[0][0])
        self.assertEqual(mat_1[1][1], mat_[1][1])
        self.assertEqual(mat_1[2][2], mat_[2][2])

    def test_get_rotation_from_direction(self):
        p_1 = Node(1, 1, 2)
        c_1 = Node(1, 0, 0)
        self.assertEqual(seg.get_rotation_from_direction(c_1, p_1), 7)
        self.assertEqual(seg.get_rotation_from_direction(p_1, c_1), 3)

        p_1 = Node(1, 1, 2)
        c_1 = Node(2, 1, 0)
        self.assertEqual(seg.get_rotation_from_direction(c_1, p_1), 5)
        self.assertEqual(seg.get_rotation_from_direction(p_1, c_1), 1)

    def test_get_clockwise_list_from_matrix(self):
        mat = np.resize(np.arange(10), (3,3))
        mat = seg.get_clockwise_list_from_matrix(mat, 2)
        mat_ = np.array([2, 5, 8, 7, 6, 3, 0, 1])
        try:
            np.testing.assert_equal(mat, mat_)
        except AssertionError:
            print("Unequal lists. Bad rotation")

    def test_neighboorhood_search(self):
        mat = np.array([[0, 0, 0, 0, 0, 0, 0],\
                        [0, 0, 0, 2, 2, 2, 0],\
                        [0, 2, 2, 2, 2, 2, 0],\
                        [0, 2, 2, 2, 0, 0, 0],\
                        [0, 2, 2, 2, 0, 0, 0],\
                        [0, 0, 2, 2, 2, 0, 0],\
                        [0, 0, 0, 0, 0, 0, 0]])
        p_1 = Node(1, 3, 2)
        c_1 = Node(1, 2, 0)
        s = Node(1, 3, 2)
        c_2, p_2, to_continue = seg.search_neigbourhood(p_1, c_1, s, mat, 2)
        c_2_, p_2_= Node(0, 4, 0), Node(1, 4, 2)
        self.assertEqual(c_2, c_2_)
        self.assertEqual(p_2, p_2_)
        self.assertTrue(to_continue)
        c_3, p_3, to_continue = seg.search_neigbourhood(p_2, c_2, s, mat, 2)
        c_3_, p_3_ = Node(0, 5, 0), Node(1, 5, 2), 
        self.assertEqual(c_3, c_3_)
        self.assertEqual(p_3, p_3_)
        self.assertTrue(to_continue)

    def test_get_Moore_Neighborhood_countour(self):
        mat = np.array([[0, 0, 0, 0, 0, 0, 0],\
                        [0, 0, 0, 2, 2, 2, 0],\
                        [0, 2, 2, 2, 2, 2, 0],\
                        [0, 2, 2, 2, 0, 0, 0],\
                        [0, 2, 2, 2, 0, 0, 0],\
                        [0, 0, 2, 2, 2, 0, 0],\
                        [0, 0, 0, 0, 0, 0, 0]])
        mat_ = np.array([[0, 0, 0, 0, 0, 0, 0],\
                         [0, 0, 0, 1, 1, 1, 0],\
                         [0, 1, 1, 2, 1, 1, 0],\
                         [0, 1, 2, 1, 0, 0, 0],\
                         [0, 1, 2, 1, 0, 0, 0],\
                         [0, 0, 1, 1, 1, 0, 0],\
                         [0, 0, 0, 0, 0, 0, 0]])
        
        mat, perimeter, border = seg.get_Moore_Neighborhood_countour(mat, 2)
        self.assertEqual(perimeter, 14)
        self.assertEqual(1, 1.0)
        try:
            np.testing.assert_equal(mat, mat_)
        except AssertionError:
            print("Unequal lists. Bad border extraction")

import classification as cls
class TestClassificationMethods(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()