Noise Filtration
================

This module carries out noise filtration over multiple tiff blocks. 


One way to reduce the effect of photobleaching on SOFI analysis is to divide a long video
into multiple blocks with fewer frames. In each block, the decrease in fluorescence intensity 
is small enough so that users canomit the effect of photobleaching. Each block can be 
considered as an individual `PysofiData` object. After the reconstruction step, all 
reconstructed images are saved in separate object, and can be put together for the filtration.

Functions
---------
.. automodule:: functions.filtering
   :members:
   :undoc-members:
   :show-inheritance:


Example
-------
A one-dimensional Gaussian mask (kernel) is first generated, and passed to the filtration along 
with an array of images in sequence that needs to be filtered (m_set):

::

    nf = masks.gauss1D_mask(shape=(1,21), sigma=2)
    m_filtered = filtering.noise_filter1d(dset, m_set, nf, filenames=filenames, return_option=True)