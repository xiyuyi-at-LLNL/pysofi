# if the image is too large and get `--NotebookApp.iopub_data_rate_limit` 
# message, you can change the setting by typing 
# "jupyter notebook --NotebookApp.iopub_data_rate_limit=10000000" 
# (or bigger number) in command line.

import numpy as np
#import cv2
from bokeh.resources import INLINE
import bokeh.io
bokeh.io.output_notebook(INLINE)
from scipy.io import loadmat
from matplotlib import colors
from matplotlib import cm
import matplotlib.pyplot as plt
import tifffile as tiff
from bokeh.plotting import figure, output_file, show, output_notebook

def ensure_positive(data):
    """
    Make sure data is positive and has no zeros.
    """
    data = data.copy()
    data[data <= 0] = np.finfo(float).eps
    return data

def bokeh_visualization(image, palette = None, save_option = False, 
                        filename = 'Image', imshow_same = True):
    '''
    Show interactive grayscale image with Bokeh in Jupyter Notebook.

    Parameters
    ----------
    image: ndarray
        The grayscale image array for visualization.
    palette: str
        Name of the Bokeh palettes. 
        The default palette is 'pink' from matplotlib.
        For a complete palette list in bokeh documentation: 
            https://docs.bokeh.org/en/latest/docs/reference/palettes.html
    save_option: bool
        Whether to save the Bokeh image as a .html file.
    filename: str 
        Name of the .html file.
    imshow_same: bool
        Whether to reversse y-axis to show the same image as using matplotlib.

    Examples
    --------
    TODO: Please refer to the demo Jupyter Notebook ''.

    Notes
    -----
    For more information about interactive visualization library Bokeh: 
    https://docs.bokeh.org/en/latest/index.html.
    '''
    image = ensure_positive(image)
    xdim, ydim = np.shape(image)
    TOOLTIPS = [("x", "$x{int}"), ("y", "$y{int}"), ("value", "@image")]
   
    if palette is None:
        # import 'pink' colormap from matplotlib
        pink_cmap = cm.get_cmap('pink', 256)    
        pink_cmap256 = pink_cmap(np.linspace(0, 1, 256))
        palette = tuple(colors.to_hex(i) for i in pink_cmap256)
    
    if imshow_same == True:
        p = figure(x_range=(0, xdim), y_range=(ydim, 0), tooltips=TOOLTIPS)
        flip_im = np.flipud(image)
        p.image(image=[flip_im], x=0, y=ydim, dw=xdim, dh=ydim, 
                palette=palette, level="image")
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None
    else:
        p = figure(x_range=(0, xdim), y_range=(0, ydim), tooltips=TOOLTIPS)
        p.image(image=[image], x=0, y=0, dw=xdim, dh=ydim, 
                palette=palette, level="image")
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None

    if save_option == False:
        output_notebook()
    else:
        output_file(filename+".html", title=filename)
        
    show(p, notebook_handle=True)

def bokeh_visualization_rgba(image, save_option = False, 
                             filename = 'RGBAimage', imshow_same = True):
    '''
    Show interactive RGBA image with Bokeh in Jupyter Notebook.

    Parameters
    ----------
    image: ndarray
        The RGBA image array for visualization.
    save_option: bool
        Whether to save the Bokeh image as a .html file.
    filename: str 
        Name of the .html file.
    imshow_same: bool
        Whether to reversse y-axis to show the same image as using matplotlib.

    Examples
    --------
    TODO: Please refer to the demo Jupyter Notebook ''.

    Notes
    -----
    For more information about interactive visualization library Bokeh: 
    https://docs.bokeh.org/en/latest/index.html.
    '''    
    image = ensure_positive(image)
    image[image>255] = 255
    assert (np.sum(image != np.uint8(image)) == 0), \
        "Converting image dtype to uint8"
    image = np.uint8(image)     
    xdim, ydim, ch = np.shape(image)
    TOOLTIPS = [("x", "$x{int}"), ("y", "$y{int}"), ("value", "@image")]
    
    if imshow_same == True:
        p = figure(x_range=(0, xdim), y_range=(ydim, 0), tooltips=TOOLTIPS)
        flip_im = np.flipud(image)
        p.image_rgba(image=[flip_im], x=0, y=ydim, dw=xdim, dh=ydim)
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None
    else:
        p = figure(x_range=(0, xdim), y_range=(0, ydim), tooltips=TOOLTIPS)
        p.image_rgba(image=[image], x=0, y=0, dw=xdim, dh=ydim)
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None

    if save_option == False:
        output_notebook()
    else:
        output_file(filename+".html", title=filename)
        
    show(p, notebook_handle=True)
    
def ind2cmap(im, cmap = 'pink'):
    '''
    Convert a grayscale image array to a rgba color image((0,1) float).
    '''
    norm_im = (im - np.min(im)) / (np.max(im) - np.min(im))
    im_cmap = cm.get_cmap(cmap) 
    color_im = im_cmap(norm_im)
    return color_im

def save_png(im, cmap='pink', filename='out.png', display_contrast=1):
    '''
    Save a image with a specific colormap. 
    Please refer to the demo Jupyter Notebook for examples.
    '''
    im = ensure_positive(im)
    color_img = ind2cmap(im, cmap)
    color_img = color_img * 255 * display_contrast
    color_img[color_img>255] = 255
    color_img = np.uint8(color_img)
    plt.imsave(filename, color_img)
    
# def save_avi(vid_array, cmap='pink', filename ='out.avi', display_contrast=1):
#     '''
#     Save a image stack with a specific colormap.
#     Please refer to the demo Jupyter Notebook for examples.
#     '''
#     frame, xdim, ydim = np.shape(vid_array)
#     out = cv2.VideoWriter(filename,cv2.VideoWriter_fourcc('M','J','P','G'),
#                           frame, (xdim,ydim))
#     for i in range(frame):
#         img = ensure_positive(vid_array[i])
#         color_img = ind2cmap(img, cmap)
#         color_img = color_img * 255 * display_contrast
#         color_img[color_img>255] = 255
#         color_img = np.uint8(color_img)
#         out.write(color_img)
#     cv2.destroyAllWindows()
#     out.release()
       

def enhance_contrast(im, cmap = 'pink', contrast_order = 1):
    '''Enhance contrast of the input image.'''
    im = ensure_positive(im)
    color_img = ind2cmap(img, cmap)
    color_img = color_img * 255 * display_contrast
    color_img[color_img>255] = 255
    color_img = np.uint8(color_img)
    return color_img