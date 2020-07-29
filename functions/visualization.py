# if the image is too large and get `--NotebookApp.iopub_data_rate_limit` message, you can change the setting by
# "jupyter notebook --NotebookApp.iopub_data_rate_limit=10000000" (or bigger number) in command line.

import numpy as np
from bokeh.resources import INLINE
import bokeh.io
bokeh.io.output_notebook(INLINE)

from bokeh.plotting import figure, output_file, show, output_notebook

def bokeh_visualization(image, palette="Inferno256", save_option = False, filename = 'Image', imshow_same = True):
    xdim, ydim = np.shape(image)
    TOOLTIPS = [("x", "$x{int}"), ("y", "$y{int}"), ("value", "@image")]
    
    if imshow_same == True:
        p = figure(x_range=(0, xdim), y_range=(ydim, 0), tooltips=TOOLTIPS)
        flip_im = np.flipud(image)
        p.image(image=[flip_im], x=0, y=ydim, dw=xdim, dh=ydim, palette=palette, level="image")
    else:
        p = figure(x_range=(0, xdim), y_range=(0, ydim), tooltips=TOOLTIPS)
        p.image(image=[image], x=0, y=0, dw=xdim, dh=ydim, palette=palette, level="image")

    if save_option == False:
        output_notebook()
    else:
        output_file(filename+".html", title=filename)
        
    show(p, notebook_handle=True)

