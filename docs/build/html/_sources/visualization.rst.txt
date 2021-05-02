Result visualizations
==============================

This module provides multiple options to visualize result images. For instance, the
image can be demonstrated interactively with Bokeh, with adjusted contrast to better
visualize detailed features, and with a transparency map.

Functions
---------
.. automodule:: functions.visualization
   :members:
   :undoc-members:
   :show-inheritance:


Examples
--------
Plot a single grayscale image interactively:

::

    visaulization.bokeh_visualization(d.get_frame(0), palette=’pink’, save_option=False, imshow_same=True)


Plot multiple grayscale images side-by-side:

::

    visualization.bokeh_visualization_mult([d.get_frame(0), d.average_image()], title_lst=[’Frame #1’, ’Average image’])


Plot a single RGBA image:

::

    visaulization.bokeh_visualization_rgb(im_4d, save_option=False, imshow_same=True)


Enhance the contrast of the input image, and plot it with Bokeh:

::

    en_im = visualization.enhance_contrast(d.ave, display_contrast=1.4)
    visualization.bokeh_visualization_rgba(en_im)
    

Image visualization with a transparency map:

::

    trans_im = add_transmap(im, trans_map=d.average_image(), cmap='pink')