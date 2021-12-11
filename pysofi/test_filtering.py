import sys
sys.path.append('../src/')

import unittest
from ddt import ddt, data, unpack
import filtering, masks
import numpy as np


@ddt
class TestFiltering(unittest.TestCase):
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

    @data({'time_series': np.append(np.arange(1,21), np.arange(21,0,-1)), 'noise_filter': masks.gauss1d_mask((1, 21), 2), 'filtered': np.array([1.38078275, 2.15655102, 3.05330466, 4.0148171, 5.00332504, 6.00059712, 7.00008513, 8.00000948, 9.00000074, 10, 11, 11.99999851,12.99998104,13.99982974,14.99880576, 15.99334993, 16.97036579, 17.89339068, 18.68689797, 19.2384345, 19.43790566, 19.2384345, 18.68689797, 17.89339068, 16.97036579, 15.99334993, 14.99880576, 13.99982974, 12.99998104, 11.99999851, 11, 10, 9.00000074, 8.00000948, 7.00008513, 6.00059712, 5.00332504, 4.0148171, 3.05330466, 2.15655102, 1.38078275])}) 
    @unpack
    def test_filter1d_same(self, time_series, noise_filter, filtered):
        calculatedResult = filtering.filter1d_same(time_series, noise_filter)
        self.assertArrayAlmostEqual(calculatedResult, filtered)

    @data({'ori_signal': np.append(np.arange(1,21), np.arange(21,0,-1)), 'kernel_size': 5, 'filtered': np.array([2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 20, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 2])}) 
    @unpack
    def test_med_smooth(self, ori_signal, kernel_size, filtered):
        calculatedResult = filtering.med_smooth(ori_signal, kernel_size)
        self.assertArrayAlmostEqual(calculatedResult, filtered)

if __name__ == '__main__':
    unittest.main(verbosity=2)