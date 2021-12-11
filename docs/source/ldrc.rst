Local Dynamic Range Compression (LDRC) Method
=============================================

This module provides an option to compress the high dynamic range of high-order 
SOFI-reconstructed images. 

High-order cumulant or moment reconstructions produce images with a large dynamic 
range for pixel intensities, making detailed features obscure. To overcome the problem,
the ldrc algorithm rescales pixel intensities of high-order reconstructions based on a 
reference image. The compression is performed locally in a small window that scans 
across the original image. In each window, the pixel intensities of the original image 
are linearly rescaled so that they have the same dynamic range as the reference window.

Functions
---------
.. automodule:: pysofi.ldrc
    :members:


Example
-------
The reference image (second-order moment reconstruction), the input image that needs 
dynamic range compression, and the scanning window size is passed to the ldrc function:

::

    ldrc_im = ldrc.ldrc(mask_im=d.moment_image(order=2), input_im=d.filtered, window_size=[25, 25])