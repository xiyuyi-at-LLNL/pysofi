import sys
sys.path.append('../src/')

import unittest
from ddt import ddt, data, unpack
import reconstruction
import numpy as np


@ddt
class TestReconstruction(unittest.TestCase):
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

    def assertNestedDictAlmostEqual(self, calculatedDictionary, expectedDictionary, places=4):
        self.assertEqual(calculatedDictionary.keys(), expectedDictionary.keys())
        for key in calculatedDictionary.keys():
            self.assertNumericDictAlmostEqual(calculatedDictionary[key], expectedDictionary[key], places=4)


    @data({'filepath': '../sampledata', 'filename': 'test_vid.tif', 'frames': [0,5], 'ave_im': [[0,2],[3,5]]}) 
    @unpack
    def test_average_image(self, filepath, filename, frames, ave_im):
        calculatedResult = reconstruction.average_image(filepath, filename, frames)
        self.assertNestedArrayEqual(calculatedResult, ave_im)

    @data({'filepath': '../sampledata', 'filename': 'test_vid.tif', 'factor': 2, 'ave_im_wfinterp': [[0,1,2],[1.5,2.5,3.5],[3,4,5]]}) 
    @unpack
    def test_average_image_with_finterp(self, filepath, filename, factor, ave_im_wfinterp):
        calculatedResult = reconstruction.average_image_with_finterp(filepath, filename, factor)
        self.assertNestedArrayEqual(calculatedResult, ave_im_wfinterp)

    @data({'filepath': '../sampledata', 'filename': 'test_vid.tif', 'order': 3, 'frames': [0,5], 'moment_im': [[0,6],[-6,0]]}) 
    @unpack
    def test_moment_im(self, filepath, filename, order, frames, moment_im):
        calculatedResult = reconstruction.calc_moment_im(filepath, filename, order, frames)
        self.assertNestedArrayEqual(calculatedResult, moment_im)

    @data({'filepath': '../sampledata', 'filename': 'test_vid.tif', 'order': 3, 'interp_num': 2, 'frames': [0,5], 'moment_im_wfinterp': [[0,0,6],[0,0,0],[-6,0,0]]}) 
    @unpack
    def test_moment_im_with_finterp(self, filepath, filename, order, interp_num, frames, moment_im_wfinterp):
        calculatedResult = reconstruction.moment_im_with_finterp(filepath, filename, order, interp_num, frames)
        self.assertNestedArrayEqual(calculatedResult, moment_im_wfinterp)

    @data({'filepath': '../sampledata', 'filename': 'test_vid.tif', 'highest_order': 4, 'moment_set': {1:[[0,0],[0,0]], 2:[[0,6],[6,0]], 3:[[0,6],[-6,0]], 4:[[0,42],[42,0]]}}) 
    @unpack
    def test_calc_moments(self, filepath, filename, highest_order, moment_set):
        calculatedResult = reconstruction.calc_moments(filepath, filename, highest_order)
        self.assertNumericDictAlmostEqual(calculatedResult, moment_set)

    @data({'moments_set': {1:[[0,0],[0,0]], 2:[[0,6],[6,0]], 3:[[0,6],[-6,0]], 4:[[0,42],[42,0]]}, 'k_set': {1:[[0,0],[0,0]], 2:[[0,6],[6,0]], 3:[[0,6],[-6,0]], 4:[[0,-66],[-66,0]]}}) 
    @unpack
    def test_calc_cumulants_from_moments(self, moments_set, k_set):
        calculatedResult = reconstruction.calc_cumulants_from_moments(moments_set)
        self.assertNumericDictAlmostEqual(calculatedResult, k_set)

    @data({'filepath': '../sampledata', 'filename': 'test_vid.tif', 'total_signal': [10,10,10,10,10]}) 
    @unpack
    def test_calc_total_signal(self, filepath, filename, total_signal):
        calculatedResult = reconstruction.calc_total_signal(filepath, filename)
        self.assertArrayAlmostEqual(calculatedResult, total_signal) 

    @data({'signal_level': np.arange(100,0,-1), 'fbc': 0.2, 'bounds': [100,80,60,40,20,1], 'frame_lst': [0,21,41,61,81,100]}) 
    @unpack
    def test_cut_frames(self, signal_level, fbc, bounds, frame_lst):
        calculated_bounds, calculated_frames = reconstruction.cut_frames(signal_level, fbc)
        self.assertArrayAlmostEqual(calculated_bounds, bounds)
        self.assertArrayAlmostEqual(calculated_frames, frame_lst) 

    @data({'filepath': '../sampledata', 'filename': 'test_vid.tif', 'frames': [0,5], 'min_im': [[0,0],[0,5]]}) 
    @unpack
    def test_min_image(self, filepath, filename, frames, min_im):
        calculatedResult = reconstruction.min_image(filepath, filename, frames)
        self.assertNestedArrayEqual(calculatedResult, min_im)

    @data({'filepath': '../sampledata', 'filename': 'test_vid2.tif', 'highest_order': 4, 'smooth_kernel': 21, 'fbc': 0.2, 'm_all': {0: {1: [[0, 0],[0, 0]], 2: [[0,47],[191,0]], 3: [[0, 0],[0, 0]], 4: [[0,4123],[65971,0]]}, 1: {1: [[0, 0],[0, 0]], 2: [[0,26],[107,0]], 3: [[0, 0],[0, 0]], 4: [[0,1298],[20779,0]]}, 2: {1: [[0, 0],[0, 0]], 2: [[0,26],[107,0]], 3: [[0, 0],[0, 0]], 4: [[0,1298],[20779,0]]}, 3: {1: [[0, 0],[0, 0]], 2: [[0,26],[107,0]], 3: [[0, 0],[0, 0]], 4: [[0,1298],[20779,0]]}, 4: {1: [[0, 0],[0, 0]], 2: [[0,44],[176,0]], 3: [[0,0],[0,0]], 4: [[0,3476],[55616,0]]}}}) 
    @unpack
    def test_moments_all_blocks(self, filepath, filename, highest_order, smooth_kernel, fbc, m_all):
        calculatedResult = reconstruction.moments_all_blocks(filepath, filename, highest_order, smooth_kernel, fbc)
        self.assertNestedDictAlmostEqual(calculatedResult, m_all)

    @data({'m_all': {0: {1: [[0, 0],[0, 0]], 2: [[0,47],[191,0]], 3: [[0, 0],[0, 0]], 4: [[0,4123],[65971,0]]}, 1: {1: [[0, 0],[0, 0]], 2: [[0,26],[107,0]], 3: [[0, 0],[0, 0]], 4: [[0,1298],[20779,0]]}, 2: {1: [[0, 0],[0, 0]], 2: [[0,26],[107,0]], 3: [[0, 0],[0, 0]], 4: [[0,1298],[20779,0]]}, 3: {1: [[0, 0],[0, 0]], 2: [[0,26],[107,  0]], 3: [[0, 0],[0, 0]], 4: [[0,1298],[20779,0]]}, 4: {1: [[0, 0],[0, 0]], 2: [[0,44],[176,0]], 3: [[0,0],[0,0]], 4: [[0,3476],[55616,0]]}}, 'k_all': {0: {1: [[0, 0],[0, 0]], 2: [[  0, 47],[191,0]], 3: [[0, 0],[0, 0]], 4: [[0,-2504],[-43472,0]]}, 1: {1:[[0, 0],[0, 0]], 2: [[0,26],[107,0]], 3: [[0, 0],[0, 0]], 4: [[0,-730],[-13568,0.]]}, 2: {1: [[0, 0],[0, 0]], 2: [[0,26],[107,0]], 3: [[0, 0],[0, 0]], 4: [[0,-730],[-13568,0]]}, 3: {1: [[0, 0],[0, 0]], 2: [[0,26],[107,0]], 3: [[0, 0],[0, 0]], 4: [[0,-730],[-13568,0]]}, 4: {1: [[0, 0],[0, 0]], 2: [[0,44],[176,0]], 3: [[0,0],[0,0]], 4: [[0,-2332],[-37312,0]]}}}) 
    @unpack
    def test_cumulants_all_blocks(self, m_all, k_all):
        calculatedResult = reconstruction.cumulants_all_blocks(m_all)
        self.assertNestedDictAlmostEqual(calculatedResult, k_all)

    @data({'filepath': '../sampledata', 'filename': 'test_vid2.tif', 'highest_order': 4, 'smooth_kernel': 21, 'fbc': 0.2, 'ave_moments':{1: [[0, 0],[0, 0]], 2: [[0,33.8],[137.6,0]], 3: [[0, 0],[0, 0]], 4: [[0, 2298.6],[36784.8,0]]}}) 
    @unpack
    def test_block_ave_moments(self, filepath, filename, highest_order, smooth_kernel, fbc, ave_moments):
        calculatedResult = reconstruction.block_ave_moments(filepath, filename, highest_order, smooth_kernel, fbc)
        self.assertNumericDictAlmostEqual(calculatedResult, ave_moments)


    @data({'filepath': '../sampledata', 'filename': 'test_vid2.tif', 'highest_order': 4, 'smooth_kernel': 21, 'fbc': 0.2, 'ave_cumulants':{1: [[0, 0],[0, 0]], 2: [[0,33.8],[137.6,0]], 3: [[0, 0],[0, 0]], 4: [[0, -1405.2],[-24297.6,0]]}}) 
    @unpack
    def test_block_ave_cumulants(self, filepath, filename, highest_order, smooth_kernel, fbc, ave_cumulants):
        calculatedResult = reconstruction.block_ave_cumulants(filepath, filename, highest_order, smooth_kernel, fbc)
        self.assertNumericDictAlmostEqual(calculatedResult, ave_cumulants)

    @data({'filepath': '../sampledata', 'filename': 'test_vid2.tif', 'tauSeries': [0,1,2], 'm_set': {2: {(0, 1): [[0,816.66666667],[3266.66666667,0]]}, 3: {(0, 1, 2): [[0, 0],[0,0]]}}}) 
    @unpack
    def test_calc_moments_with_lag(self, filepath, filename, tauSeries, m_set):
        calculatedResult = reconstruction.calc_moments_with_lag(filepath, filename, tauSeries)
        self.assertNestedDictAlmostEqual(calculatedResult, m_set)

    @data({'m_set': {2: {(0, 1): [[0,816.66666667],[3266.66666667,0]]}, 3: {(0, 1, 2): [[0, 0],[0,0]]}}, 'tauSeries': [0,1,2], 'k_set': {2: [[0,816.66666667], [3266.66666667, 0]], 3: [[0, 0],[0,0]]}}) 
    @unpack
    def test_calc_cumulants_with_lag(self, m_set, tauSeries, k_set):
        calculatedResult = reconstruction.calc_cumulants_from_moments_with_lag(m_set, tauSeries)
        self.assertNumericDictAlmostEqual(calculatedResult, k_set) 


if __name__ == '__main__':
    unittest.main(verbosity=2)