Getting Started
================

.. finish after construct pypi
.. https://epidemicsonnetworks.readthedocs.io/en/latest/GettingStarted.html#installation

pysofi can be installed as a standard python package either via `conda`
or `pip` (later). pysofi runs on OS X, Windows and Linux.

.. For updates on the latest FRETBursts version please refer to the
.. :doc:`Release Notes (What's new?) <releasenotes>`.

.. _package_install:

Installation
-------------
pysofi requires installation of `Anaconda <https://docs.anaconda.com/anaconda/install/>`__.

The latest version of pysofi can cloned or downloaded from 
https://github.com/xiyuyi-at-LLNL/pysofi.

Before running pysofi, create the environment by running the following code 
in Anaconda prompt or the mac terminal:

::

    conda env create -f pysofi.yml

Then activate the environment with:

::

    conda activate pysofi

Before running notebook files, set up the environment of Ipython kernel with 
the following code:

::

    ipython kernel install --user --name=pysofi

After starting jupyter notebook, switch kernel to current environment by clicking 
"Kernel -> Change kernel -> pysofi" from the jupyter notebook dropdown menu.

.. _pysofi_overview:

pysofi Overview
----------------
pysofi implements three primary collection of SOFI analysis routines, including the shared process 
that enables the traditional SOFI analysis, the SOFI 2.0 analysis routines and a a separate SOFI 
relevant routine named Multi Order Cumulant Analysis (MOCA) to demonstrate the modular extension 
of pysofi for additional analysis method.

Specifically, one can load input data (.tiff format) or partially processed data (such as .tiff 
movie with bleaching correction and/or .with Fourier interpolation) into the `pysofiData` class, 
and perform bleaching correction, Fourier interpolation, moments calculations and cumulant 
calculations in various sequences. 

For SOFI 2.0 routine, one can continue from the previous processing 
step where moments are obtained or directly loaded from pre-computed results, and perform follow 
up analysis of the first noise filtering, deconvolution, the second noise filtering followed with 
a final local dynamic range compression processing to obtain the SOFI 2.0 results.

Likewise, the MOCA module can continue from cumulantcalculations or direclty load pre-computed 
cumulants, and perform the processing step to produce spatial brigthness distribution and 
blinking statistics distribution. Detailed implementation examples of the analysis routines will 
be discussed in the paper in accompany with tutorials implemented Jupyter notebooks (see demos). 


Sample Data
-----------
All simulated and experimental data are shared on figshare: https://figshare.com/s/47d97a2df930380c96bb