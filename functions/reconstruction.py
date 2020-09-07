from . import finterp as f
import numpy as np
import tifffile as tiff
import scipy.special
import sys


def average_image(filepath, filename):
    """
    Get the average image for a video file (tiff stack).
    """
    imstack = tiff.TiffFile(filepath + '/' + filename)
    xdim, ydim = np.shape(imstack.pages[0])
    mvlength = len(imstack.pages)
    mean_im = np.zeros((xdim, ydim))

    for frame_num in range(mvlength):
        im = tiff.imread(filepath + '/' + filename, key=frame_num)
        mean_im = mean_im + im

    mean_im = mean_im / mvlength

    return mean_im


def average_image_with_finterp(filepath, filename, interp_num):
    """
    Get the average image with fourier interpolation for a video file.
    Parameters
    ----------
    filepath : str
        Path to the tiff file.
    filename : str
        Name of the tiff file.
    interp_num : int
        Interpolation factor.

    Returns
    -------
    mean_im : ndarray
        The average image after fourier interpolation. Interpolated
    images can be further used for SOFI processing.
    """
    original_mean_im = average_image(filepath, filename)
    finterp_mean_im = f.fourier_interp_array(original_mean_im, [interp_num])
    return finterp_mean_im[0]


def calc_moment_im(filepath, filename, order, mvlength=0, mean_im=None):
    """
    Get one moment-reconstructed image of a defined order for a video file.

    Parameters
    ----------
    filepath : str
        Path to the tiff file.
    filename : str
        Name of the tiff file.
    order : int
        The order number of moment-reconstructed image.
    mvlength : int
        The length of video for the reconstruction.
    mean_im : ndarray
        Average image of the tiff stack.
    Returns
    -------
    moment_im : ndarray
        The moments-reconstructed image.
    """
    if mean_im is None:
        mean_im = average_image(filepath, filename)
    imstack = tiff.TiffFile(filepath + '/' + filename)
    xdim, ydim = np.shape(imstack.pages[0])
    if mvlength == 0:
        mvlength = len(imstack.pages)
    moment_im = np.zeros((xdim, ydim))
    for frame_num in range(mvlength):
        im = tiff.imread(filepath + '/' + filename, key=frame_num)
        moment_im = moment_im + (im - mean_im)**order
        sys.stdout.write('\r')
        sys.stdout.write("[{:{}}] {:.1f}%".format(
            "="*int(30/(mvlength-1)*frame_num), 29,
            (100/(mvlength-1)*frame_num)))
        sys.stdout.flush()
    moment_im = np.int64(moment_im / mvlength)
    return moment_im


def moment_im_with_finterp(filepath, filename, order, interp_num,
                           mvlength=0, mean_im=None):
    """
    Get one moment-reconstructed image of a defined order for a video file.

    Parameters
    ----------
    filepath : str
        Path to the tiff file.
    filename : str
        Name of the tiff file.
    order : int
        The order number of moment-reconstructed image.
    interp_num : int
        The interpolation factor.
    mvlength : int
        The length of video for the reconstruction.
    mean_im : ndarray
        Average image of the tiff stack.

    Returns
    -------
    moment_im : ndarray
        The moments-reconstructed image.
    """
    if mean_im is None:
        mean_im = average_image_with_finterp(filepath, filename, interp_num)

    imstack = tiff.TiffFile(filepath + '/' + filename)
    xdim, ydim = np.shape(imstack.pages[0])
    if mvlength == 0:
        mvlength = len(imstack.pages)
    moment_im = np.zeros(((xdim-1)*interp_num+1, (ydim-1)*interp_num+1))
    for frame_num in range(mvlength):
        im = tiff.imread(filepath + '/' + filename, key=frame_num)
        interp_im = f.fourier_interp_array(im, [interp_num])[0]
        moment_im = moment_im + (interp_im - mean_im)**order
        sys.stdout.write('\r')
        sys.stdout.write("[{:{}}] {:.1f}%".format(
            "="*int(30/(mvlength-1)*frame_num), 29,
            (100/(mvlength-1)*frame_num)))
        sys.stdout.flush()
    moment_im = np.int64(moment_im / mvlength)
    return moment_im


def calc_moments(filepath, filename, highest_order, m_set={}, mean_im=None):
    """
    Get all moment-reconstructed images to the user-defined highest order for
    a video file(tiff stack).

    Parameters
    ----------
    filepath : str
        Path to the tiff file.
    filename : str
        Name of the tiff file.
    highest_order : int
        The highest order number of moment-reconstructed images.
    m_set : dict
        order number (int) -> image (ndarray)
        A dictionary of previously calcualted moment-reconstructed images.
    mean_im : ndarray
        Average image of the tiff stack.

    Returns
    -------
    m_set : dict
        order number (int) -> image (ndarray)
        A dictionary of calcualted moment-reconstructed images.
    """
    if m_set:
        current_order = max(m_set.keys())
    else:
        current_order = 0

    if mean_im is None:
        mean_im = average_image(filepath, filename)

    imstack = tiff.TiffFile(filepath + '/' + filename)
    xdim, ydim = np.shape(imstack.pages[0])
    mvlength = len(imstack.pages)

    # print out the progress
    def ordinal(n): return "%d%s" % (
        n, "tsnrhtdd"[(n//10 % 10 != 1)*(n % 10 < 4)*n % 10::4])
    order_lst = [ordinal(n+1) for n in range(highest_order)]

    if highest_order > current_order:
        for order in range(current_order, highest_order):
            print('Calculating the %s-order moment reconstruction...' %
                  order_lst[order])
            m_set[order+1] = np.zeros((xdim, ydim))
            for frame_num in range(mvlength):
                im = tiff.imread(filepath + '/' + filename, key=frame_num)
                m_set[order+1] = m_set[order+1] + \
                    np.power(im - mean_im, order+1)
                sys.stdout.write('\r')
                sys.stdout.write("[{:{}}] {:.1f}%".format(
                    "="*int(30/(mvlength-1)*frame_num), 29,
                    (100/(mvlength-1)*frame_num)))
                sys.stdout.flush()
            m_set[order+1] = np.int64(m_set[order+1] / mvlength)
            print('\n')
    return m_set


def calc_cumulants_from_moments(moment_set):
    """
    Calculate cumulant-reconstructed images from moment-reconstructed images.

    Parameters
    ----------
    moment_set : dict
        order number (int) -> image (ndarray)
        A dictionary of calcualted moment-reconstructed images.
    Returns
    -------
    k_set : dict
        order number (int) -> image (ndarray)
        A dictionary of calcualted cumulant-reconstructed images.
    """
    if moment_set == {}:
        raise Exception("'moment_set' is empty.")

    k_set = {}
    highest_order = max(moment_set.keys())
    for order in range(1, highest_order + 1):
        k_set[order] = moment_set[order] - \
            np.sum(np.array([scipy.special.comb(order-1, i) * k_set[order-i] *
                             moment_set[i] for i in range(1, order)]), axis=0)

    return k_set
