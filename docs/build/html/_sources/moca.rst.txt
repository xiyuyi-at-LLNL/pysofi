Multi-Order Cumulant Analysis (MOCA)
=====================================

This module helps to perform MOCA on the measurements and extract photophysical properties
of the emitters.

Multi-order cumulant analysis (MOCA) is based on an analytical relationship between 
multi-order SOFI cumulants and photophysical parameters of emitters (blinking on-time ratio 
and on-time brightness), and locally search for the mostlikely spatial distribution of these
parameters. For more detailed information on MOCA theory, please refer to the paper.


Functions
---------
.. automodule:: pysofi.moca
   :members:
   :undoc-members:
   :show-inheritance:


Example
-------
To calculate moment-reconstructed images, on-time ratio map and brightness map:

::

   ac, rho_map, eps_map = moca.moca(filename, filepath, tauSeries=[0,0,0,0,0,0,0],frames=[0,1000], mask_dim=(301,301), res=1000)


To add a transparency map and plot the on-time ratio map:

::

   rho_map_color = visualization.add_transmap(rho_map, ac[2], 'cool')
   plt.imshow(rho_map_color, cmap='cool')