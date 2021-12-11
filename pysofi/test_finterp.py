import sys
sys.path.append('../src/')

import unittest
from ddt import ddt, data, unpack
import finterp
import numpy as np


@ddt
class TestFinterp(unittest.TestCase):
    def assertNestedArrayEqual(self, calculatedNestedArray, expectedNestedArray, places=4):
        for i in range(len(calculatedNestedArray)):
            for j in range(len(calculatedNestedArray[i])):
                if isinstance(expectedNestedArray[i][j], complex):
                    self.assertAlmostEqual(calculatedNestedArray[i][j].real, expectedNestedArray[i][j].real, places=4)
                    self.assertAlmostEqual(calculatedNestedArray[i][j].imag, expectedNestedArray[i][j].imag, places=4)
                else:
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

    @data({'xrange': 4, 'yrange': 4, 'fx': np.array([[1,1,1,1], [1,0-1.j,-1,0+1.j], [1,-1,1,-1], [ 1,0+1.j,-1,0-1.j]]), 'fy': np.array([[1,1,1,1], [1,0-1.j,-1,0+1.j], [1,-1,1,-1], [ 1,0+1.j,-1,0-1.j]])}) 
    @unpack
    def test_ft_matrix2d(self, xrange, yrange, fx, fy):
        calculatedFx, calculatedFy = finterp.ft_matrix2d(xrange, yrange)
        self.assertNestedArrayEqual(calculatedFx, fx)
        self.assertNestedArrayEqual(calculatedFy, fy)

    @data({'xrange': 4, 'yrange': 4, 'interp_num': 3, 'ifx': np.array([[0.25, 0.25, 0.25, 0.25], [0.25, 0.21650635+0.125j, 0.125-0.21650635j, 0.21650635-0.125j], [0.25, 0.125+0.21650635j, -0.125-0.21650635j, 0.125-0.21650635j], [0.25, 0+0.25j, -0.25, 0-0.25j], [0.25, -0.125+0.21650635j, -0.125+0.21650635j, -0.125-0.21650635j],[0.25, -0.21650635+0.125j, 0.125+0.21650635j, -0.21650635-0.125j], [0.25, -0.25, 0.25, -0.25],[0.25, -0.21650635-0.125j, 0.125-0.21650635j, -0.21650635+0.125j], [0.25, -0.125   -0.21650635j, -0.125-0.21650635j, -0.125+0.21650635j], [0.25, -0-0.25j, -0.25, 0+0.25j]]), 'ify': np.array([[0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25], [0.25, 0.21650635+0.125j, 0.125+0.21650635j, 0+0.25j, -0.125+0.21650635j, -0.21650635+0.125j, -0.25, -0.21650635-0.125j, -0.125-0.21650635j, -0-0.25j], [0.25, 0.125-0.21650635j, -0.125-0.21650635j, -0.25, -0.125+0.21650635j, 0.125+0.21650635j, 0.25, 0.125-0.21650635j, -0.125-0.21650635j, -0.25],[0.25, 0.21650635-0.125j, 0.125-0.21650635j, 0-0.25j, -0.125-0.21650635j, -0.21650635-0.125j, -0.25, -0.21650635+0.125j, -0.125+0.21650635j,0+0.25j]])}) 
    @unpack
    def test_ift_matrix2d(self, xrange, yrange, interp_num, ifx, ify):
        calculatediFx, calculatediFy = finterp.ift_matrix2d(xrange, yrange, interp_num)
        self.assertNestedArrayEqual(calculatediFx, ifx)
        self.assertNestedArrayEqual(calculatediFy, ify)

    @data({'xrange': 4, 'yrange': 4, 'interp_num': 3, 'interp_im': np.array([[0.01831564, 0.01831564, 0.01831564, 0.01831564], [0.01831564, 0.01831564, 0.01831564, 0.01831564], [0.01831564, 0.01831564, 0.01831564, 0.01831564], [0.01831564, 0.01831564, 0.01831564, 0.01831564]])}) 
    @unpack
    def test_interpolate_image(self, xrange, yrange, interp_num, interp_im):
        x, y = np.meshgrid(np.linspace(-1, 1, xrange/2), np.linspace(-1, 1, xrange/2))
        im = np.exp(-((np.sqrt(x*x + y*y))**2 / (2*0.5**2)))
        fx, fy = finterp.ft_matrix2d(xrange, yrange)
        ifx, ify = finterp.ift_matrix2d(xrange, yrange, interp_num)
        calculatedResult = finterp.interpolate_image(im, fx, fy, ifx, ify, interp_num)
        self.assertNestedArrayEqual(calculatedResult, interp_im)
       
    @data({'im': np.array([[0,1],[1,2],[3,4]]), 'interp_num_lst': [2,3], 'interp_im_lst': [np.array([[0, 0.5, 1], [0.300641, 0.800641, 1.30064], [1, 1.5, 2], [2.032692, 2.53269, 3.03269],[3, 3.5, 4]]), np.array([[0, 0.316987, 0.6830127, 1], [0.1621098, 0.479097, 0.84512251, 1.1621098], [0.4855889, 0.8025762, 1.1686016, 1.4855889], [1, 1.316987, 1.6830127, 2], [1.670381, 1.9873687, 2.353394, 2.670381], [2.388791, 2.7057787, 3.0718041, 3.388791], [3, 3.316987, 3.6830127, 4]])]}) 
    @unpack
    def test_fourier_interp_array(self, im, interp_num_lst,interp_im_lst):
        calculatedResult = finterp.fourier_interp_array(im, interp_num_lst)
        for i in range(len(interp_num_lst)):
            self.assertNestedArrayEqual(calculatedResult[i], interp_im_lst[i])  

    @data({'filepath': '../sampledata', 'filename': 'test_vid', 'factor': [3], 'frames': [2,4], 'save_option': False, 'return_option': True, 'interp_imstack_lst': [[np.array([[0,0,0,0],[2,2,2,2],[3,3,3,3],[5,5,5,5]]), np.array([[0,2,3,5],[0,2,3,5],[0,2,3,5],[0,2,3,5]])]]}) 
    @unpack
    def test_fourier_interp_tiff(self, filepath, filename, factor, frames, save_option, return_option, interp_imstack_lst):
        calculatedResult = finterp.fourier_interp_tiff(filepath, filename, factor, frames, save_option, return_option)
        for i in range(frames[1]-frames[0]):
            self.assertNestedArrayEqual(calculatedResult[0][i], interp_imstack_lst[0][i])

if __name__ == '__main__':
    unittest.main(verbosity=2)