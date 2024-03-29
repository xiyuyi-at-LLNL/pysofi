Getting Started
================

.. finish after construct pypi
.. https://epidemicsonnetworks.readthedocs.io/en/latest/GettingStarted.html#installation

PySOFI can be used as a standard python package. We are currently working on a `pip` release.
PySOFI is tested on OS (MacOS Mojave, 10.14.16), Windows (Windows 10) and Linux (the high performance computer from Livermore Computing).

.. For updates on the latest FRETBursts version please refer to the
.. :doc:`Release Notes (What's new?) <releasenotes>`.

.. _package_install:

Installation
-------------
PySOFI requires installation of `Anaconda <https://docs.anaconda.com/anaconda/install/>`__.

The latest version of PySOFI can cloned or downloaded from
https://github.com/xiyuyi-at-LLNL/pysofi.

It can also be acquired through pip install:

::

    pip install pysofi

Before running PySOFI, create the environment by running the following code
in Anaconda prompt or the mac terminal:

::

    conda env create -f env_MacOS_Majave.yml #(tested for MacOS Majave 10.14.3)

or

::

    cconda env create -f env_Win10.yml #(tested for Windows 10)

or

::

    conda env create -f env_linux.yml #(tested for Linux kernal version of 3.10.0-1160.45.1.1chaos.ch6.x86_64)


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

PySOFI Overview
----------------
PySOFI implements three primary collection of SOFI analysis routines, including the shared process
that enables the traditional SOFI analysis, the SOFI 2.0 analysis routines and a a separate SOFI 
relevant routine named Multi Order Cumulant Analysis (MOCA, non-peer reviewed) to demonstrate the 
modular extension of PySOFI for additional analysis method.

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