Reconstruction Options
=======================

This module provides multiple image processing and reconstruction options. For instance,
calculate the average image, calculate one moment-reconstructed image of a defined order, 
calculate all moment-reconstructed or cumulant reconstructed images to the user-defined 
highest order, calculate average moment or cumulant reconstructions over multiple user-
defind blocks, with or without Fourier interpolation or bleaching correction.


Functions
---------
.. automodule:: pysofi.reconstruction
   :members:
   :undoc-members:
   :show-inheritance:


Examples
--------
To calculate fourth-order moment-reconstructed image using frame 50 to 100 of the input video
with FOurier interpolation (factor=2):

::

   m_im = reconstruction.moment_im_with_finterp(filepath, filename, order=4, interp_num=2, frames=[50,100])


To calculate up to the fourth-order cumulant-reconstructed image:

::

   m_set = reconstruction.calc_moments(filepath, filename, highest_order=4)
   k_set = reconstruction.calc_cumulants_from_moments(m_set)


To calculate the average moment reconstruction by dividing the whol video into 25 blocks, and in each block the 
signal decrease is 4% of the total signal decrease:

::

   mean_mim = reconstruction.block_ave_moments(filepath, filename, 4, smooth_kernel=251, fbc=0.04)


To perform bleaching correction on a input video and save the corrected images in a new tiff file:

::

   reconstruction.correct_bleaching(filepath, filename, fbc=0.04, smooth_kernel=251,
                                    save_option=True, return_option=False)




