"""
This file runs the tests for all pysofi methods. To run this test, put 'test_vid.tif' in the folder '../sampledata' 
and this file in the same folder as pysofi.py, and run the command ‘python -m unittest discover -s .’ 

"""
import sys
sys.path.append('../src/')

import unittest
from ddt import ddt, data, unpack
import masks #change to file name
import pysofi
import numpy as np


@ddt
class TestPysofi(unittest.TestCase):
    def assertNestedArrayEqual(self, calculatedNestedArray, expectedNestedArray, places=4):
        for i in range(len(calculatedNestedArray)):
            for j in range(len(calculatedNestedArray[i])):
                self.assertAlmostEqual(calculatedNestedArray[i][j], expectedNestedArray[i][j], places=4)

    def assertArrayAlmostEqual(self, calculatedArray, expectedArray, places=4):
        for i in range(len(calculatedArray)):
            self.assertAlmostEqual(calculatedArray[i], expectedArray[i], places=4)
    
    def assertNumericDictAlmostEqual(self, calculatedDictionary, expectedDictionary, places=4):
        self.assertEqual(calculatedDictionary.keys(), expectedDictionary.keys())
        for key in calculatedDictionary.keys():
            self.assertNestedArrayEqual(calculatedDictionary[key], expectedDictionary[key], places=4)

    @data({'filepath': '../sampledata', 'filename': 'test_vid.tif', 'ave_im': [[0,2],[3,5]]}) 
    @unpack
    def test_average_image(self, filepath, filename, ave_im):
        d = pysofi.PysofiData(filepath, filename)
        calculatedResult = d.average_image()
        self.assertNestedArrayEqual(calculatedResult, ave_im)

    @data({'filepath': '../sampledata', 'filename': 'test_vid.tif', 'factor': 2, 'ave_im_wfinterp': [[0,1,2],[1.5,2.5,3.5],[3,4,5]]}) 
    @unpack
    def test_average_image_wfinterp(self, filepath, filename, factor, ave_im_wfinterp):
        d = pysofi.PysofiData(filepath, filename)
        calculatedResult = d.average_image_with_finterp(factor)
        self.assertNestedArrayEqual(calculatedResult, ave_im_wfinterp)

    @data({'filepath': '../sampledata', 'filename': 'test_vid.tif', 'order': 3, 'moment_im': [[0,6],[-6,0]]}) 
    @unpack
    def test_moment_im(self, filepath, filename, order, moment_im):
        d = pysofi.PysofiData(filepath, filename)
        calculatedResult = d.moment_image(order)
        self.assertNestedArrayEqual(calculatedResult, moment_im)

    @data({'filepath': '../sampledata', 'filename': 'test_vid.tif', 'order': 4, 'moment_set': {1:[[0,0],[0,0]], 2:[[0,6],[6,0]], 3:[[0,6],[-6,0]], 4:[[0,42],[42,0]]}}) 
    @unpack
    def test_moment_set(self, filepath, filename, order, moment_set):
        d = pysofi.PysofiData(filepath, filename)
        calculatedResult = d.calc_moments_set(order)
        self.assertNumericDictAlmostEqual(calculatedResult, moment_set)

    @data({'filepath': '../sampledata', 'filename': 'test_vid.tif', 'order': 4, 'cumulant_set': {1:[[0,0],[0,0]], 2:[[0,6],[6,0]], 3:[[0,6],[-6,0]], 4:[[0,-66],[-66,0]]}}) 
    @unpack
    def test_cumulant_set(self, filepath, filename, order, cumulant_set):
        d = pysofi.PysofiData(filepath, filename)
        calculatedResult = d.cumulants_images(order)
        self.assertNumericDictAlmostEqual(calculatedResult, cumulant_set)

    @data({'filepath': '../sampledata', 'filename': 'test_vid.tif', 'window_size': [1,2], 'order': 4, 'ldrc': [[2,0],[0,5]]}) 
    @unpack
    def test_ldrc(self, filepath, filename, window_size, order, ldrc):
        d = pysofi.PysofiData(filepath, filename)
        mask_im = d.average_image()
        input_im = d.cumulants_images(order)[order]
        calculatedResult = d.ldrc(order, window_size, mask_im, input_im)
        self.assertNestedArrayEqual(calculatedResult, ldrc)

    @data({'filepath': '../sampledata', 'filename': 'test_vid.tif', 'est_psf': masks.gauss2d_mask((1,1),2), 'order': 4, 'deconv_lambda': 1.5, 'deconv_iter': 2, 'deconvsk': [[0,0],[0,0]]}) 
    @unpack
    def test_deconvsk(self, filepath, filename, est_psf, order, deconv_lambda, deconv_iter, deconvsk):
        d = pysofi.PysofiData(filepath, filename)
        input_im = d.cumulants_images(order)[order]
        calculatedResult = d.deconvsk(est_psf, input_im, deconv_lambda, deconv_iter)
        self.assertNestedArrayEqual(calculatedResult, deconvsk)

    @data({'filepath': '../sampledata', 'filename': 'test_vid.tif', 'interp_lst': [2,3], 'finterp': [[[0,1,2],[1.5,2.5,3.5],[3,4,5]],[[0,0.63397,1.36602,2],[0.95096,1.584936,2.31698,2.95096],[2.049038,2.68301,3.41506,4.04903],[3,3.63397,4.36602,5]]]}) 
    @unpack
    def test_finterp_im(self, filepath, filename, interp_lst, finterp):
        d = pysofi.PysofiData(filepath, filename)
        input_im = d.average_image()
        calculatedResult = d.finterp_image(input_im, interp_lst)
        for i in range(len(interp_lst)):
            self.assertNestedArrayEqual(calculatedResult[i], finterp[i])

    @data({'filepath': '../sampledata', 'filename': 'test_vid.tif', 'dims': (5,2,2)}) 
    @unpack
    def test_getdims(self, filepath, filename, dims):
        d = pysofi.PysofiData(filepath, filename)
        calculatedResult = d.get_dims()
        self.assertArrayAlmostEqual(calculatedResult, dims)

    @data({'filepath': '../sampledata', 'filename': 'test_vid.tif', 'frame_num': 1, 'frame': [[0,0],[5,5]]}) 
    @unpack
    def test_getframe(self, filepath, filename, frame_num, frame):
        d = pysofi.PysofiData(filepath, filename)
        calculatedResult = d.get_frame(frame_num).tolist()
        self.assertNestedArrayEqual(calculatedResult, frame)


if __name__ == '__main__':
    unittest.main(verbosity=2)

