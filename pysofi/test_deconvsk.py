import sys
sys.path.append('../src/')

import unittest
from ddt import ddt, data, unpack
import deconvsk
import numpy as np


@ddt
class TestDeconvsk(unittest.TestCase):
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

    def imagedirac(size):
        array = np.zeros((size, size))
        array[size // 2, size // 2] = 1
        return array


    @data({'image': np.array([[1,2],[3,4]]), 'shape': (4,4), 'position': 'center', 'pad_im': np.array([[0,0,0,0],[0,1,2,0],[0,3,4,0],[0,0,0,0]])}) 
    @unpack
    def test_zero_pad(self, image, shape, position, pad_im):
        calculatedResult = deconvsk.zero_pad(image, shape, position)
        self.assertNestedArrayEqual(calculatedResult, pad_im)

    @data({'psf': imagedirac(4), 'shape': [4,4], 'otf': np.ones((4,4))}) 
    @unpack
    def test_psf2otf(self, psf, shape, otf):
        calculatedResult = deconvsk.psf2otf(psf, shape)
        self.assertNestedArrayEqual(calculatedResult, otf)

    @data({'otf': np.ones((4,4)), 'shape': [4,4], 'psf': imagedirac(4)}) 
    @unpack
    def test_otf2psf(self, otf, shape, psf):
        calculatedResult = deconvsk.otf2psf(otf, shape)
        self.assertNestedArrayEqual(calculatedResult, psf)
        
    @data({'image': imagedirac(4), 'h': np.ones((4,4)), 'f': np.array([[1,-1,1,-1],[-1,1,-1,1],[1,-1,1,-1],[-1,1,-1,1]])}) 
    @unpack
    def test_corelucy(self, image, h, f):
        calculatedResult = deconvsk.corelucy(image, h)
        self.assertNestedArrayEqual(calculatedResult, f) 

    @data({'image': imagedirac(4), 'psf': np.ones((4,4)), 'iterations': 15, 'new_psf': np.array([[0,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,0]]), 'deconv_im': np.array([[0,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,0]])}) 
    @unpack
    def test_richardson_lucy(self, image, psf, iterations, new_psf, deconv_im):
        calculatedP, calculatedJ = deconvsk.richardson_lucy(image, psf, iterations)
        self.assertNestedArrayEqual(calculatedP, new_psf) 
        self.assertNestedArrayEqual(calculatedJ, deconv_im) 

    @data({'est_psf': np.ones((4,4)), 'input_im':imagedirac(4), 'deconv_lambda': 1.5, 'deconv_iter': 15, 'deconv_im': np.array([[0,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,0]])}) 
    @unpack
    def test_deconvsk(self, est_psf, input_im, deconv_lambda, deconv_iter, deconv_im):
        calculatedResult = deconvsk.deconvsk(est_psf, input_im, deconv_lambda, deconv_iter)
        self.assertNestedArrayEqual(calculatedResult, deconv_im) 


if __name__ == '__main__':
    unittest.main(verbosity=2)