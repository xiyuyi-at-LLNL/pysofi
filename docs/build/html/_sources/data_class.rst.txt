.. currentmodule:: functions.data

The "Data()" class
==================

The :class:`Data` class is the main container for smFRET measurements.
It contains timestamps, detectors and all the results of data processing
such as background estimation, burst data, fitted FRET and so on.

It can be loaded xxx.

.. contents::


Summary information
-------------------
List of :class:`Data` attributes and
methods providing summary information on the measurement:

.. class:: Data

    .. autoattribute:: filename

    .. autoattribute:: filepath

    .. autoattribute:: ave

    .. autoattribute:: finterp_factor

    .. automethod:: average_image


Analysis methods
----------------
.. automodule:: functions.data
    :members:
