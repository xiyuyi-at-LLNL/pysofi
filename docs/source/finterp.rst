Fourier Interpolation Module
============================

Functions
---------
.. automodule:: functions.finterp
   :members:
   :undoc-members:
   :show-inheritance:


Examples
--------
To interpolate a single image and generate two images with different interpolation factors (4 and 6):

::

   im_lst = finterp.fourier_interp_array(im, [4,6])


To insert seven new pixels between adjacent physical pixels for the first 1000 frames of the input video 
and save the interpolated images in a new tiff video.

::

   finterp.fourier_interp_tiff(filepath, filename, interp_num_lst=[8], frames=[0,1000], 
                               save_option=True, return_option=False):