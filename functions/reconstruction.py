from . import finterp as f
from . import filtering
import numpy as np
import tifffile as tiff
import scipy.special
import sys
import collections
from scipy import signal


def average_image(filepath, filename, frames=[]):
    """
    Get the average image for a video file (tiff stack), either for the
    whole video or for user defined frames. 

    Parameters
    ----------
    filepath : str
        Path to the tiff file.
    filename : str
        Name of the tiff file.
    frames : list of int, optional
        Start and end frame number if not the whole video is used.

    Returns
    -------
    mean_im : ndarray
        The average image.    
    """
    imstack = tiff.TiffFile(filepath + '/' + filename)
    xdim, ydim = np.shape(imstack.pages[0])
    mvlength = len(imstack.pages)
    mean_im = np.zeros((xdim, ydim))

    if frames:
        for frame_num in range(frames[0], frames[1]):
            im = tiff.imread(filepath + '/' + filename, key=frame_num)
            mean_im = mean_im + im
        mean_im = mean_im / (frames[1] - frames[0])
    else:
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


def calc_block_moments(filepath, filename, highest_order, frames=[]):
    """
    Get moment-reconstructed images for user-defined frames (block) of
    a video file(tiff stack). 

    Parameters
    ----------
    filepath : str
        Path to the tiff file.
    filename : str
        Name of the tiff file.
    highest_order : int
        The highest order number of moment-reconstructed images.
    frames : list of int
        Start and end frame number.

    Returns
    -------
    m_set : dict
        order number (int) -> image (ndarray)
        A dictionary of calcualted moment-reconstructed images.

    Notes
    -----
    Similar to 'calc_moments'. Here we omit previously calculated m_set 
    and mean_im as inputs since a block usually has much fewer number of
    frames and takes shorter calculation time.
    """
    mean_im = average_image(filepath, filename, frames)
    imstack = tiff.TiffFile(filepath + '/' + filename)
    xdim, ydim = np.shape(imstack.pages[0])
    block_length = frames[1]-frames[0]
    m_set = {}

    for order in range(highest_order):
        m_set[order+1] = np.zeros((xdim, ydim))
        for frame_num in range(frames[0], frames[1]):
            im = tiff.imread(filepath + '/' + filename, key=frame_num)
            m_set[order+1] = m_set[order+1] + \
                np.power(im - mean_im, order+1)
        m_set[order+1] = np.int64(m_set[order+1] / block_length)
        sys.stdout.write('\r')
        sys.stdout.write("[{:{}}] {:.1f}%".format(
            "="*int(30/(highest_order-1)*order), 29,
            (100/(highest_order-1)*order)))
        sys.stdout.flush()
    print('\n')
    return m_set


def calc_total_signal(filepath, filename):
    """
    Calculate the total signal intensity of each frame for the whole 
    video (tiff stack).

    Parameters
    ----------
    filepath : str
        Path to the tiff file.
    filename : str
        Name of the tiff file.

    Returns
    -------
    total_signal : 1darray
        Signal intensity of each frame ordered by the frame number.
    """
    imstack = tiff.TiffFile(filepath + '/' + filename)
    mvlength = len(imstack.pages)
    total_signal = np.zeros(mvlength)

    for frame_num in range(mvlength):
        im = tiff.imread(filepath + '/' + filename, key=frame_num)
        total_signal[frame_num] = sum(sum(im))
    return total_signal


def cut_frames(signal_level, fbc=0.04):
    """
    Find the list of frame number to cut the whole signal plot into seperate
    blocks based on the change of total signal intensity. 

    Parameters
    ----------
    signal_level : 1darray
        Signal change over time (can be derived from 'calc_total_signal').
    fbc : float
        The fraction of signal decrease within each block compared to the
        total signal decrease.

    Returns
    -------
    bounds : list of int
        Signal intensities on the boundary of each block.
    frame_lst : list of int
        Frame number where to cut the whole signal plot into blocks.

    Notes
    -----
    The number of blocks is the inverse of the bleaching correction factor, 
    fbc. For instance, if fbc=0.04, it means that in each block, the 
    decrease in signal intensity is 4% if the total decrease, and the whole
    plot / video is cut into 25 blocks. For some data, it is possible that 
    the maximun signal intensity does not appear at the beginning of the 
    signal plot. Here, we consider all frames before the maximum in the 
    same block as the maximum frame / intensity since usually the number 
    of frames is not too large. The user can add in extra blocks if the 
    initial intensity is much smaller than the maximum.
    """
    max_intensity, min_intensity = np.max(signal_level), np.min(signal_level)
    frame_num = np.argmax(signal_level)
    total_diff = max_intensity - min_intensity
    block_num = int(1/fbc)
    frame_lst = []
    # lower bound of intensity for each block
    bounds = [int(max_intensity-total_diff*i*fbc)
              for i in range(1, block_num+1)]
    i = 0
    while frame_num < len(signal_level) and i <= block_num:
        if signal_level[frame_num] < bounds[i]:
            frame_lst.append(frame_num)
            frame_num += 1
            i += 1
        else:
            frame_num += 1
    frame_lst = [0] + frame_lst + [len(signal_level)]
    bounds = [int(max_intensity)] + bounds
    return bounds, frame_lst


def moments_all_blocks(filepath, filename,
                       highest_order, smooth_kernel=251, fbc=0.04):
    """
    Get moment-reconstructed images for seperate blocks with user-defined
    noise filtering and bleaching correction factor (fbc).

    Within each block, the amount of signal decrease is identical. This 
    amount of signal decrease is characterized by the bleaching correction 
    factor, fbc, which is the fractional signal decrease within each block
    (as compared to the total signal decrease over the whole signal plot). 

    Parameters
    ----------
    filepath : str
        Path to the tiff file.
    filename : str
        Name of the tiff file.
    highest_order : int
        The highest order number of moment-reconstructed images.
    smooth_kernel : int
        The size of the median filter ('filtering.med_smooth') window.
    fbc : float
        The fraction of signal decrease within each block compared to the
        total signal decrease.

    Returns
    -------
    m_set_all_blocks : dict
        block_number (int) -> {order number (int) -> image (ndarray)}
        A dictionary of moment-reconstructed images of each block.

    Notes
    -----
    Similar to 'calc_moments'. Here we omit previously calculated m_set 
    and mean_im as inputs since a block usually has much fewer number of
    frames and takes shorter calculation time.
    """
    all_signal = calc_total_signal(filepath, filename)
    filtered_signal = filtering.med_smooth(all_signal, smooth_kernel)
    _, cut_frame = cut_frames(filtered_signal, fbc)
    block_num = int(1/fbc)
    m_set_all_blocks = {}
    for i in range(block_num):
        print('Calculating moments of block %d...' % i)
        m_set_all_blocks[i] = calc_block_moments(filepath,
                                                 filename,
                                                 highest_order,
                                                 [cut_frame[i],
                                                  cut_frame[i+1]])

    return m_set_all_blocks


def cumulants_all_blocks(m_set_all_blocks):
    """
    Calculate cumulant-reconstructed images from moment-reconstructed images
    of each block. Similar to 'calc_cumulants_from_moments'.
    """
    if m_set_all_blocks == {}:
        raise Exception("'moment_set' is empty.")

    k_set_all_blocks = {i: calc_cumulants_from_moments(m_set_all_blocks[i])
                        for i in m_set_all_blocks}
    return k_set_all_blocks


def block_ave_cumulants(filepath, filename,
                        highest_order, smooth_kernel=251, fbc=0.04):
    """
    Get average cumulant-reconstructed images of all blocks determined by 
    user-defined noise filtering and bleaching correction factor (fbc).

    Within each block, the amount of signal decrease is identical. This 
    amount of signal decrease is characterized by the bleaching correction 
    factor, fbc, which is the fractional signal decrease within each block
    (as compared to the total signal decrease over the whole signal plot). 

    Parameters
    ----------
    filepath : str
        Path to the tiff file.
    filename : str
        Name of the tiff file.
    highest_order : int
        The highest order number of moment-reconstructed images.
    smooth_kernel : int
        The size of the median filter ('filtering.med_smooth') window.
    fbc : float
        The fraction of signal decrease within each block compared to the
        total signal decrease.

    Returns
    -------
    ave_k_set : dict
        order number (int) -> image (ndarray)
        A dictionary of avereage cumulant-reconstructed images of all blocks.

    Notes
    -----
    For more information on noise filtering and bleaching corrextion, please
    see appendix 3 of [1].

    References
    ----------
    .. [1] X. Yi, and S. Weiss. "Cusp-artifacts in high order superresolution 
    optical fluctuation imaging." bioRxiv: 545574 (2019).
    """
    k_set_all_blocks = cumulants_all_blocks(
        moments_all_blocks(filepath, filename,
                           highest_order,
                           smooth_kernel, fbc))
    block_num = len(k_set_all_blocks)
    k_set_lst = list(k_set_all_blocks.values())
    counter = collections.Counter()
    for d in k_set_lst:
        counter.update(d)
    ave_k_set = dict(counter)
    ave_k_set = {i: ave_k_set[i]/block_num for i in ave_k_set}
    return ave_k_set


def block_ave_moments(filepath, filename,
                      highest_order, smooth_kernel=251, fbc=0.04):
    """
    Get average moment-reconstructed images of all blocks determined by 
    user-defined noise filtering and bleaching correction factor (fbc).
    Similar to 'block_ave_cumulants'.

    Parameters
    ----------
    filepath : str
        Path to the tiff file.
    filename : str
        Name of the tiff file.
    highest_order : int
        The highest order number of moment-reconstructed images.
    smooth_kernel : int
        The size of the median filter ('filtering.med_smooth') window.
    fbc : float
        The fraction of signal decrease within each block compared to the
        total signal decrease.

    Returns
    -------
    ave_m_set : dict
        order number (int) -> image (ndarray)
        A dictionary of avereage moment-reconstructed images of all blocks.
    """
    m_set_all_blocks = moments_all_blocks(filepath, filename,
                                          highest_order, smooth_kernel, fbc)
    block_num = len(m_set_all_blocks)
    m_set_lst = list(m_set_all_blocks.values())
    counter = collections.Counter()
    for d in m_set_lst:
        counter.update(d)
    ave_m_set = dict(counter)
    ave_m_set = {i: ave_m_set[i]/block_num for i in ave_m_set}
    return ave_m_set
