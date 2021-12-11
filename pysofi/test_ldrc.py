import sys
sys.path.append('../src/')

import unittest
from ddt import ddt, data, unpack
import ldrc
import numpy as np


@ddt
class TestLDRC(unittest.TestCase):
    def assertNestedArrayEqual(self, calculatedNestedArray, expectedNestedArray, places=4):
        for i in range(len(calculatedNestedArray)):
            for j in range(len(calculatedNestedArray[i])):
                self.assertAlmostEqual(calculatedNestedArray[i][j], expectedNestedArray[i][j], places=4)


    @data({'mask_im': np.ones((5,8)), 'input_im': np.append(np.arange(0,20), np.arange(21,1,-1)).reshape(5,8), 'order': 1, 'window_size': [5,5], 'ldrc_im':np.array([[0, 0.02380952, 0.0484127 , 0.08703008, 0.13775063, 0.17192982, 0.21052632, 0.26315789], [0.38095238, 0.41428571, 0.44908104, 0.49279449, 0.54351504, 0.58596491, 0.63157895, 0.68421053], [0.76190476, 0.8047619 , 0.84974937, 0.8985589 , 1, 0.94824561, 0.89473684, 0.84210526], [0.80952381, 0.75595238, 0.69949875, 0.64495614, 0.59423559, 0.53421053, 0.47368421, 0.42105263], [0.42857143, 0.36547619, 0.29883041, 0.23919173, 0.18847118, 0.12017544, 0.05263158, 0]])})
    @unpack
    def test_ldrc(self, mask_im, input_im, order, window_size, ldrc_im):
        calculatedResult = ldrc.ldrc(mask_im, input_im, order, window_size)
        self.assertNestedArrayEqual(calculatedResult, ldrc_im) 

if __name__ == '__main__':
    unittest.main(verbosity=2)