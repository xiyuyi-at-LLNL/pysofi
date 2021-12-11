# if the image is too large and get `--NotebookApp.iopub_data_rate_limit`
# message, you can change the setting by typing
# "jupyter notebook --NotebookApp.iopub_data_rate_limit=10000000"
# (or bigger number) in command line.

import numpy as np
from . import switches as s
if s.SPHINX_SWITCH is False:
    import cv2

from bokeh.resources import INLINE
from bokeh.layouts import row
import bokeh.io
from scipy.io import loadmat
from matplotlib import colors
from matplotlib import cm
import matplotlib.pyplot as plt
from . import switches as s
if s.SPHINX_SWITCH is False:
    import tifffile as tiff

from bokeh.plotting import figure, output_file, show, output_notebook
from bokeh.models import ColumnDataSource
from  bokeh.models import PanTool,ResetTool,BoxZoomTool
bokeh.io.output_notebook(INLINE)
import copy

def ensure_positive(data):
    """
    Make sure data is positive and has no zeros.
    """
    data = data.copy()
    data[data <= 0] = np.finfo(float).eps
    return data


def bokeh_visualization(image, palette=None, save_option=False,
                        filename='Image', imshow_same=True):
    """
    Show interactive grayscale image with Bokeh in Jupyter Notebook.
    Parameters
    ----------
    image : ndarray
        The grayscale image array for visualization.
    palette : str
        Name of the Bokeh palettes. 
        The default palette is 'pink' from matplotlib.
        For a complete palette list in bokeh documentation: 
            https://docs.bokeh.org/en/latest/docs/reference/palettes.html
    save_option : bool
        Whether to save the Bokeh image as a .html file.
    @@ -49,39 +54,70 @@ def bokeh_visualization(image, palette=None, save_option=False,
    Notes
    -----
    For more information about interactive visualization library Bokeh: 
    https://docs.bokeh.org/en/latest/index.html.
    """
    image = ensure_positive(image)
    xdim, ydim = np.shape(image)
    TOOLTIPS = [("x", "$x{int}"), ("y", "$y{int}"), ("value", "@image")]

    if palette is None:
        # import 'pink' colormap from matplotlib
        pink_cmap = cm.get_cmap('pink', 256)
        pink_cmap256 = pink_cmap(np.linspace(0, 1, 256))
        palette = tuple(colors.to_hex(i) for i in pink_cmap256)

    if imshow_same is True:
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

    if save_option is False:
        output_notebook()
    else:
        output_file(filename+".html", title=filename)

    show(p, notebook_handle=True)


def bokeh_visualization_rgba(image, save_option=False,
                             filename='RGBAimage', imshow_same=True, saturation_factor=1):
    """
    Show interactive RGBA image with Bokeh in Jupyter Notebook.

    Parameters
    ----------
    image : ndarray
        The RGBA image array for visualization.
    save_option : bool
        Whether to save the Bokeh image as a .html file.
    filename : str
        Name of the .html file.
    imshow_same : bool
        Whether to reversse y-axis to show the same image as using matplotlib.

    Notes
    -----
    For more information about interactive visualization library Bokeh: 
    https://docs.bokeh.org/en/latest/index.html.
    """
    image = ensure_positive(image)*saturation_factor
    image[np.where(image>255)] = 255
    # assert (np.sum(image != np.uint8(image)) == 0), \
    #     "Converting image dtype to uint8"
    image = np.uint8(image)
    xdim, ydim, ch = np.shape(image)
    TOOLTIPS = [("x", "$x{int}"), ("y", "$y{int}"), ("value", "@image")]

    # make an image_for_rgba that can be used to display the input matrix 'image' using bokeh image_rgba
    image_for_rgba = np.uint32(np.ones(image[:, :, 0].shape))
    view = image_for_rgba.view(dtype=np.uint8).reshape((image.shape[0], image.shape[1], 4))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            view[i, j, 0] = image[i, j, 0]
            view[i, j, 1] = image[i, j, 1]
            view[i, j, 2] = image[i, j, 2]
            view[i, j, 3] = image[i, j, 3]

    if imshow_same is True:
        p = figure(x_range=(0, xdim), y_range=(ydim, 0), tooltips=TOOLTIPS)
        flip_im = np.flipud(image_for_rgba)
        p.image_rgba(image=[flip_im], x=0, y=ydim, dw=xdim, dh=ydim)
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None
    else:
        p = figure(x_range=(0, xdim), y_range=(0, ydim), tooltips=TOOLTIPS)
        p.image_rgba(image=[image_for_rgba], x=0, y=0, dw=xdim, dh=ydim)
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None

    if save_option is False:
        output_notebook()
    else:
        output_file(filename+".html", title=filename)

    show(p, notebook_handle=True)


def bokeh_visualization_mult(image_lst, title_lst=None, palette=None,
                             save_option=False, filename='Image',
                             imshow_same=True):
    """
    Show interactive grayscale image with Bokeh in Jupyter Notebook.
    Parameters
    ----------
    image_lst : list(ndarray)
        A list of grayscale images array for visualization and comparison.
    title_lst : list of str
        A list of titles for images plotted in Bokeh.
    palette : str
        Name of the Bokeh palettes.
        The default palette is 'pink' from matplotlib.
        For a complete palette list in bokeh documentation:
            https://docs.bokeh.org/en/latest/docs/reference/palettes.html
    save_option : bool
        Whether to save the Bokeh image as a .html file.
    @@ -49,39 +54,70 @@ def bokeh_visualization(image, palette=None, save_option=False,
    Notes
    -----
    For more information about interactive visualization library Bokeh:
    https://docs.bokeh.org/en/latest/index.html.
    """
    if palette is None:
        # import 'pink' colormap from matplotlib
        pink_cmap = cm.get_cmap('pink', 256)
        pink_cmap256 = pink_cmap(np.linspace(0, 1, 256))
        palette = tuple(colors.to_hex(i) for i in pink_cmap256)
    TOOLTIPS = [("x", "$x{int}"), ("y", "$y{int}"), ("value", "@image")]
    tools = [BoxZoomTool(), PanTool(), ResetTool()]

    all_im = []
    if title_lst is None:
        title_lst=np.arange(len(image_lst))
    for i in range(len(image_lst)):
    # for image, title in zip(image_lst, title_lst):
        image, title = image_lst[i], title_lst[i]
        image = ensure_positive(image)
        xdim, ydim = np.shape(image)
        if i == 0:
            if imshow_same is True:
                p1 = figure(x_range=(0, xdim), y_range=(ydim, 0), title=title,
                            plot_width=500, plot_height=500, 
                            tooltips=TOOLTIPS, tools=tools)
                flip_im = np.flipud(image)
                p1.image(image=[flip_im], x=0, y=ydim, dw=xdim, dh=ydim,
                         palette=palette, level="image")
                p1.xgrid.grid_line_color = None
                p1.ygrid.grid_line_color = None
            else:
                p1 = figure(x_range=(0, xdim), y_range=(0, ydim), title=title,
                            plot_width=500, plot_height=500, 
                            tooltips=TOOLTIPS, tools=tools)
                p1.image(image=[image], x=0, y=0, dw=xdim, dh=ydim,
                         palette=palette, level="image")
                p1.xgrid.grid_line_color = None
                p1.ygrid.grid_line_color = None
            all_im.append(p1)
        else:
            if imshow_same is True:
                p = figure(x_range=p1.x_range, y_range=p1.y_range, title=title,
                           plot_width=500, plot_height=500, 
                           tooltips=TOOLTIPS, tools=tools)
                flip_im = np.flipud(image)
                p.image(image=[flip_im], x=0, y=ydim, dw=xdim, dh=ydim,
                        palette=palette, level="image")
                p.xgrid.grid_line_color = None
                p.ygrid.grid_line_color = None
            else:
                p = figure(x_range=p1.x_range, y_range=p1.y_range, title=title,
                           plot_width=500, plot_height=500, 
                           tooltips=TOOLTIPS, tools=tools)
                p.image(image=[image], x=0, y=0, dw=xdim, dh=ydim,
                        palette=palette, level="image")
                p.xgrid.grid_line_color = None
                p.ygrid.grid_line_color = None
            all_im.append(p)            

    if save_option is False:
        output_notebook()
    else:
        output_file(filename + ".html", title=filename)

    show(row(all_im), notebook_handle=True)


def ind2cmap(im, cmap='pink'):
    """
    Convert a grayscale image array to a rgba color image((0,1) float).
    """
    norm_im = (im - np.min(im)) / (np.max(im) - np.min(im))
    im_cmap = cm.get_cmap(cmap)
    color_im = im_cmap(norm_im)
    return color_im


def set_range(im, range_set=[0, 1]):
    """
    Set display range of the input image to a given value range (norm_range).
    """
    im[im<range_set[0]] = range_set[0]
    im[im>range_set[1]] = range_set[1]
    return im


def save_png(im, cmap='pink', filename='out.png', display_contrast=1):
    """
    Save a image with a specific colormap. 
    Please refer to the demo Jupyter Notebook for examples.
    """
    im = ensure_positive(im)
    color_img = ind2cmap(im, cmap)
    color_img = color_img * 255 * display_contrast
    color_img[color_img > 255] = 255
    color_img = np.uint8(color_img)
    plt.imsave(filename, color_img)


def save_avi(vid_array, cmap='pink', filename='out.avi', display_contrast=1, os='macos'):
    """
    Save a image stack with a specific colormap. If the user is using windows,
    please set os='windows'.
    """
    frame, xdim, ydim = np.shape(vid_array)
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                          frame, (ydim, xdim))
    #out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'DIVX'),
    #                      frame, (xdim, ydim))
    for i in range(frame):
        img = ensure_positive(vid_array[i])
        color_img = ind2cmap(img, cmap)
        color_img = color_img * 255 * display_contrast
        color_img[color_img > 255] = 255
        color_img = np.uint8(color_img)
        if os == 'windows':
            color_img = color_img[:,:,0:3]
        out.write(color_img)
    out.release()
    cv2.destroyAllWindows()


def enhance_contrast(im, cmap='pink', display_contrast=1):
    """Enhance contrast of the input image."""
    im = ensure_positive(im)
    color_img = ind2cmap(im, cmap)
    color_img = color_img * 255 * display_contrast
    color_img[color_img > 255] = 255
    color_img = np.uint8(color_img)
    return color_img


def add_transmap(im, trans_map, cmap='cool'):
    """
    Prepare the image for visualization by adding a transparency map.
    It colorcodes 'im' with colormap 'cmap', and multiplied by the
    transparancy map 'trans_map'. 
    Parameters
    ----------
    im : 2darray
        A gray-scale image encoding the parameters that need to be color-
        coded.
    trans_map : 2darray
        A grey-scale mask encoding the intensity information.
    cmap : str
        A colormap provided by 'matplotlib' or defined by the user.

    Returns
    -------
    im_color : 3darray
        A RGBA image array for visualization.
    """
    im_color = ind2cmap(im, cmap)
    trans_map = (trans_map - np.min(trans_map)) / \
        (np.max(trans_map) - np.min(trans_map))
    for i in range(3):
        im_color[:,:,i] = im_color[:,:,i] * trans_map
    return im_color
