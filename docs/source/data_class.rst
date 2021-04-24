The "PysofiData()" class
========================

The `PysofiData` class is the fundamental data container that carries the measurement 
data and pysofi-related properties and behaviors. User-defined parameters and 
intermediate results are bundled to the `PysofiData` object as attributes as well 
as processing steps as methods. The full SOFI 2.0 analysis pipeline can be achieved 
based on multiple PysofiData methods. Each processing step is connected to a separate 
outside module supported by pysofi, and can be utilized not only by pysofi, but also 
by other computational imaging developers for similar purposes.

Attributes and Methods
-----------------------

.. automodule:: functions.pysofi
   :members:
   :undoc-members:
   :show-inheritance:


Load Data
---------
The simulated or experimental measurements can be loaded into the `pysofi.PysofiData` object
using:
::

    d = pysofi.PysofiData(filepath, filename)