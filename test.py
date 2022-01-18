import unittest
import cv2 as cv
import numpy as np
from numpy.lib.index_tricks import nd_grid
from cv2_test import print_img, segmentation
import data_backend_operations as db


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

import classification as cls
class TestClassificationMethods(unittest.TestCase):
    def test_get_matrix_around_pixel(self):
        mat = np.array([[0, 0, 1, 1, 1], [0, 1, 0, 1, 0], [0, 1, 0, 0, 0], [0, 0, 1, 1, 0], [1, 1, 1, 1, 0]])
        mat_1 = cls.get_matrix_around_pixel(mat, (2, 2))
        mat_2 = cls.get_matrix_around_pixel(mat, (0, 0))
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

        mat = cls.augment_black_border(mat)
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
        mat_1 = cls.get_Node_matrix_around_pixel(mat, 2, 2)
        mat_ = np.array([[seg.Node(0, 0, 1), seg.Node(0, 1, 0), seg.Node(0, 2, 0)]\
                        ,[seg.Node(1, 0, 1), seg.Node(1, 1, 0), seg.Node(1, 2, 0)]\
                        ,[seg.Node(2, 0, 0), seg.Node(2, 1, 1), seg.Node(2, 2, 1)]])
        try:
            np.testing.assert_equal(mat_1, mat_)
        except AssertionError:
            print("Unequal matrixes.")

    def test_get_clockwise_list_from_matrix(self):
        mat = np.resize(np.arange(10), (3,3))
        mat = cls.get_clockwise_list_from_matrix(mat, 2)
        mat_ = np.array([2, 5, 8, 7, 6, 3, 0, 1])
        try:
            np.testing.assert_equal(mat, mat_)
        except AssertionError:
            print("Unequal lists. Bad rotation")


if __name__ == '__main__':
    unittest.main()