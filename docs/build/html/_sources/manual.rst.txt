PySOFI Reference Manual
=======================

Introduction
------------
PySOFI consists of 8 modules, each contains a group of functions relevant to
the module, and a main data class(`pysofiData`, encapsulated in pysofi.py) that 
integrates the function steps from each module to perform SOFI analyses. Such 
design enables flexible composition of different SOFI analysis routines through 
the `pysofiData` class. Extensions of extra functionalities can be implemented 
as extra.py modules to the PySOFI package, and integrated by implementing the
relevant methods and attributes to the `pysofiData` class. 

In summary, PySOFI contains various function modules relevant to SOFI analysis:
* `reconstruction.py <https://github.com/xiyuyi-at-LLNL/pysofi/blob/master/pysofi/reconstruction.py>`__
provides capabilities for moments and SOFI cumulants calculations. 
The cumulant calculation is implemented using recursive relations and is capable of 
calculating cumulants up to arbitrary orders. It also provides functions for bleaching 
correctionof a .tiff movie necessary where bleaching of the signal level greatly 
influences the validity of SOFI cumulants. 

* `finterp.py <https://github.com/xiyuyi-at-LLNL/pysofi/blob/master/pysofi/finterp.py>`__
provides Fourier interpolation on .tiff stacks for fSOFI processing.pysofi also contains 
the modules for SOFI 2.0 calculations listes below.
 
* `filtering.py <https://github.com/xiyuyi-at-LLNL/pysofi/blob/master/pysofi/filtering.py>`__
for pixel-wise noise filtering along the time axis.

* `deconvsk.py <https://github.com/xiyuyi-at-LLNL/pysofi/blob/master/pysofi/deconvsk.py>`__
provides functions forshrinking kernel deconvolution (DeconvSK, non-peer reviewed).

* `ldrc.py <https://github.com/xiyuyi-at-LLNL/pysofi/blob/master/pysofi/ldrc.py>`__
provides local dynamic range compression (ldrc) of images with large dynamic range of pixel values.

* `visualization.py <https://github.com/xiyuyi-at-LLNL/pysofi/blob/master/pysofi/visualization.py>`__
module is implemented for interactive visualization of the results using 
`Bokeh <https://docs.bokeh.org/en/latest/index.html>`__. 

Detailed examples for each module will be discussed in our paper and on Github, each accompanied with a 
tutorial implemented in Jupyter notebook (see Demos)).


Modules
-------

.. toctree::
   :maxdepth: 2

   PysofiData class <data_class>
   Fourier Interpolation <finterp>
   Reconstructions <reconstruction>
   filter
   deconvsk
   LDRC Method <ldrc>
   moca
   visualization
   masks
   
