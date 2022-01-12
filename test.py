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

    def test_convert_RGB2GRAY(self):
        img_gray = cv.cvtColor(IMG, cv.COLOR_BGR2GRAY)
        img_gray_test = pr.convert_RGB2GRAY(IMG)
        self.assertAlmostEqual(img_gray.all(), img_gray_test.all(), places=None, delta = 1.1)
    
    def test_convert_RGB2HSV(self):
        img_hsv  = cv.cvtColor(IMG, cv.COLOR_BGR2HSV)
        img_hsv_test = pr.convert_BGR2HSV(IMG)
        self.assertAlmostEqual(img_hsv.all(), img_hsv_test.all(), places=None, delta = 1.1)

    def test_dilation(self):
        img = pr.convert_RGB2GRAY(IMG)
        _, img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
        img_dilated_test = pr.dilate_img(img, kern_size=5)
        ker = np.ones((5,5), dtype=np.uint8)
        img_dilated = cv.dilate(img, kernel=ker)
        self.assertEqual(img_dilated.all(), img_dilated_test.all())

    def test_erosion(self):
        img = pr.convert_RGB2GRAY(IMG)
        _, img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
        img_eroded_test = pr.erode_img(img, kern_size=5)
        ker = np.ones((5,5), dtype=np.uint8)
        img_eroded = cv.dilate(img, kernel=ker)
        self.assertEqual(img_eroded.all(), img_eroded_test.all())

import segmentation as seg
class TestSegmentationMethods(unittest.TestCase):
    def test_flood_fill_agorithm(self):
        mat = np.array([[0, 0, 1, 1, 1], [0, 1, 0, 1, 0], [0, 1, 0, 1, 0], [0, 0, 1, 1, 0], [1, 1, 1, 1, 0]])
        mat = seg.get_flood_fill_alg(mat, 0, 2, 2)
        mat_ = np.array([[0, 0, 2, 2, 2], [0, 1, 0, 2, 0], [0, 1, 0, 2, 0], [0, 0, 2, 2, 0], [2, 2, 2, 2, 0]])
        try:
            np.testing.assert_equal(mat, mat_, err_msg="Assertion failed")
            self.assertTrue(True)
        except AssertionError:
            print("Assertion failed")

if __name__ == '__main__':

    unittest.main()