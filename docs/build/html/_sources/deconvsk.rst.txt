Shrinking Kernel Deconvolution
==============================

This module carries out shrinking kernel deconvolution on SOFI reconstructed images. 


With the help of high-order SOFI analysis, the point spread function (PSF) of the 
optical system can be estimated. Since the acquired fluorescence image is a convolution 
between the system PSF and emitters’ locations, the true locations of emitters can be 
determined using deconvolution. In SOFI 2.0, a consecutive Richard-Lucy deconvolution 
with a series of different 2D Gaussian kernels (shrinking kernels) is applied on each 
frame of the noise filtered moment-reconstructions. This method is called the shrinking 
kernel deconvolution (deconvSK).

Functions
---------
.. automodule:: functions.deconvsk
   :members:
   :undoc-members:
   :show-inheritance:


Example
-------
Set the initial guess of the PSF (deconv_psf) as a normalized 2D Gaussian, and conduct 
deconvSK iteratively on the average image 20 times:

::

    deconv_psf = masks.gauss2d_mask((51, 51), 2)
    deconv_psf = deconv_psf / np.max(deconv_psf)
    deconv_im = deconvsk.deconvsk(est_psf=deconv_psf, input_im=d.average_image(), deconv_lambda=1.5, deconv_iter=20)