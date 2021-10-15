from . import reconstruction as rec
from . import visualization as vis
from . import masks
import numpy as np
import sys
from . import switches as s
if s.SPHINX_SWITCH is False:
    import tifffile as tiff

import matplotlib.pyplot as plt
from scipy import ndimage


def fit_sigma(filepath, filename, frames=[]):
    '''
    Use the second-order auto- and cross-cumulants to determine 
    the sigma of the gaussian psf by fitting SOFI-XC2 / SOFI-AC2 = 
    exp(-(r1-r2)**2/(4*sigma**2)).
    '''
    ri_range = 2
    series_length = ((2 * ri_range+1) ** 2 - 1) // 2
    ri_series = [(i, j) for i in range(-ri_range, 0) 
                 for j in range(-ri_range, ri_range+1)] + \
                     [(0, i) for i in range(-ri_range, 0)]
    xi2 = [((ri_series[i][0]*2)**2 + (ri_series[i][1]*2)**2)//2
           for i in range(series_length)]

    # calculate second-order auto- and cross-cumulants
    ac2 = rec.calc_moment_im(filepath, filename, 2, frames)
    xc2 = rec.calc_xc_im(filepath, filename, 2, frames)
    # crop ac2 to match the dimension with xc2
    imstack = tiff.TiffFile(filepath + '/' + filename)
    xdim, ydim = np.shape(imstack.pages[0])
    ac2_crop = ac2[ri_range:xdim-ri_range, ri_range:ydim-ri_range]
    imstack.close()
    # fit xc2 / ac2
    xc2ac2_slope = []
    for i in range(series_length):
        y = xc2[ri_series[i]].reshape(-1)
        x = ac2_crop.reshape(-1)
        w = (ac2_crop.reshape(-1))**2
        w[w == np.inf] = 0
        w[w == np.NINF] = 0
        p = np.polyfit(x, y, 1, w=w)
        if p[0] < 0:
            p[0] = np.finfo(float).eps
        xc2ac2_slope.append(p[0])
    y = np.log(xc2ac2_slope)
    p1 = np.polyfit(xi2, np.log(xc2ac2_slope), 1)[0]
    sigFit = np.sqrt(-1/2/p1)
    return sigFit


def ensure_positive(data):
    """Make sure data is positive and has no zeros."""
    data = data.copy()
    data[data <= 0] = np.finfo(float).eps
    return data


# def sorted_k_partitions(seq, k):
#     """
#     Returns a list of all unique k-partitions of `seq`.
#     Each partition is a list of parts, and each part is a tuple.
#     """
#     n = len(seq)
#     groups = []
#
#     def generate_partitions(i):
#         if i >= n:
#             yield list(map(tuple, groups))
#         else:
#             if n - i > k - len(groups):
#                 for group in groups:
#                     group.append(seq[i])
#                     yield from generate_partitions(i + 1)
#                     group.pop()
#             if len(groups) < k:
#                 groups.append([seq[i]])
#                 yield from generate_partitions(i + 1)
#                 groups.pop()
#     result = generate_partitions(0)
#
#     # Sort the parts in each partition in shortlex order
#     result = [sorted(ps, key = lambda p: (len(p), p)) for ps in result]
#     # Sort partitions by the length of each part, then lexicographically.
#     result = sorted(result, key = lambda ps: (*map(len, ps), ps))
#     # delete partitions with only 1 element
#     result = [p for p in result if len(p[0])>1]
#
#     return result
    
    
def esti_rhoeps(xn_set, res=1000):
    """
    Estimate on-time-ratio (rho_map) and brightness (eps_map) with X3 - X7
    (xn_set) by fitting and finding the minimun values difference for 5 
    functions with user defined precision (res). Xn can be directly computed
    from fluorescence signal and cross-correlation of fluorescence signal 
    from different pixels.
    Xn = ACn * U**(2n/(n-2)) / AC2, where n is the order and U is the PSF.
    """
    rho_list  = np.linspace(1/res, 1, res)
    xdim, ydim = np.shape(xn_set[3])
    rho_map = np.zeros((xdim, ydim))
    rho_tru = np.zeros((xdim, ydim))
    eps_map = np.zeros((xdim, ydim))
    # make sure there is no zero in xn_set
    xn_set = {order:xi+np.finfo('float').eps for order, xi in xn_set.items()}

    for i in range(xdim):
        for j in range(ydim):
            eps_set = {}
            eps_set[3] = 1/xn_set[3][i,j] * (1-2*rho_list)
            eps_set[4] = 1/xn_set[4][i,j]*(1-6*rho_list + 6*rho_list**2)
            eps_set[5] = 1/xn_set[5][i,j]*(1-14*rho_list + 36*rho_list**2 - \
                24*rho_list**3)
            eps_set[6] = 1/xn_set[6][i,j]*(1-30*rho_list + 150*rho_list**2 - \
                240*rho_list**3  + 120*rho_list**4)
            eps_set[7] = 1/xn_set[7][i,j]*(1-62*rho_list + 540*rho_list**2 - \
                1560*rho_list**3 + 1800*rho_list**4 - 720*rho_list**5)
            inds = np.intersect1d(
                       np.intersect1d(
                           np.intersect1d(
                               np.intersect1d(np.where(eps_set[3]>=0), 
                                              np.where(eps_set[4]>=0)), 
                           np.where(eps_set[5]>=0)), 
                       np.where(eps_set[6]>=0)), 
                   np.where(eps_set[7]>=0))
            if len(inds) > 0:
                yset = np.array([eps_set[3], 
                                 abs(eps_set[4])**(1/2), 
                                 abs(eps_set[5])**(1/3), 
                                 abs(eps_set[6])**(1/4), 
                                 abs(eps_set[7])**(1/5)])
                ydis = np.sum(yset**2,axis=0) - np.sum(yset,axis=0)**2/5
                tag = np.argmin(ydis)
                rho_map[i, j] = rho_list[tag]
                rho_tru[i, j] = 1 / np.min(ydis)
                eps_map[i, j] = 1 / np.mean(yset[:,tag])
            else:
                rho_map[i, j], rho_tru[i, j], eps_map[i, j] = 0, 0, 0
    return rho_map, rho_tru, eps_map


# def calc_block_moments(filepath, filename, highest_order, frames=[]):
#     """
#     Get moment-reconstructed images for user-defined frames (block) of
#     a video file(tiff stack).
#
#     Parameters
#     ----------
#     filepath : str
#         Path to the tiff file.
#     filename : str
#         Name of the tiff file.
#     highest_order : int
#         The highest order number of moment-reconstructed images.
#     frames : list of int
#         Start and end frame number.
#
#     Returns
#     -------
#     m_set : dict
#         order number (int) -> image (ndarray)
#         A dictionary of calcualted moment-reconstructed images.
#
#     Notes
#     -----
#     Similar to 'calc_moments'. Here we omit previously calculated m_set
#     and mean_im as inputs since a block usually has much fewer number of
#     frames and takes shorter calculation time.
#     """
#     mean_im = rec.average_image(filepath, filename, frames)
#     imstack = tiff.TiffFile(filepath + '/' + filename)
#     if not frames:
#         mvlength = len(imstack.pages)
#         frames = [0, mvlength]
#     xdim, ydim = np.shape(imstack.pages[0])
#     block_length = frames[1]-frames[0]
#     m_set = {}
#
#     for order in range(highest_order):
#         m_set[order+1] = np.zeros((xdim, ydim))
#         for frame_num in range(frames[0], frames[1]):
#             im = tiff.imread(filepath + '/' + filename, key=frame_num)
#             m_set[order+1] = m_set[order+1] + \
#                 np.power(im - mean_im, order+1)
#         m_set[order+1] = m_set[order+1] / block_length
#         sys.stdout.write('\r')
#         sys.stdout.write("[{:{}}] {:.1f}%".format(
#             "="*int(30/(highest_order-1)*order), 29,
#             (100/(highest_order-1)*order)))
#         sys.stdout.flush()
#     imstack.close()
#     print('\n')
#
#     return m_set


def moca(filename, filepath, tauSeries, frames=[], mask_dim=(301, 301), 
         res=1000):
    """
    Conduct multi-order cumulant analysis (MOCA) to extract the	photo-
    physical information (on-time-ratio rho and brightness eps) of emitters 
    at high labeling density.

    Parameters
    ----------
    filepath : str
        Path to the tiff file.
    filename : str
        Name of the tiff file.
    tauSeries : list of int
        A list of time lags for frames contribute to moment reconstruction.
        The first element is recommended to be 0, and there should be seven 
        elements in the list. This time lag is used for moments and cumulants
        calculations.
    frames : list of int
        The start and end frame number.
    mask_dim : tuple (int, int)
        Initial guess for the PSF size.
    res : int
        Resolution / precision for the fitting step 'esti_rhoeps' to get rho
        and eps for each pixel.

    Returns
    -------
    k_set : dict
        order number (int) -> image (ndarray)
        A dictionary of calcualted cumulant-reconstructed images with time 
        lags.
    rho_map : 2darray
        Fitted on-time-ratio (rho) for each pixel. It has the same dimension
        (x,y) as the original input video.
    eps_map : 2darray
        Fitted brightness (eps) for each pixel. It has the same dimension
        (x,y) as the original input video.    
    """
    if len(tauSeries) != 7:
        raise Exception("The number of time lags should be seven.")
    if tauSeries[0] != 0:
        raise Exception("The first time lag should be zero.")

    # 1. estimate sigma for Gaussian PSF using AC2 / XC2
    sigFit = fit_sigma(filepath, filename, frames)
    order = len(tauSeries)

    # 2. cumulants reconstruction
    if sum(tauSeries) == 0:
        m_set = rec.calc_block_moments(
            filepath, filename, order, frames)
        k_set = rec.calc_cumulants_from_moments(m_set)
    else:
        m_set = rec.calc_moments_with_lag(
            filepath, filename, tauSeries, frames)
        k_set = rec.calc_cumulants_from_moments_with_lag(m_set, tauSeries)
    
    # 3. re-size PSF for different orders
    xn_set = {}    
    for i in range(3, order+1):
        exp_order = 2 * i / (i - 2)
        u = masks.gauss2d_mask(mask_dim, sigFit / np.sqrt(exp_order))
        u = u / np.max(u)
        cv = ndimage.convolve(k_set[i], u)
        xn_set[i] = cv / ensure_positive(k_set[2])

    # 4. estimate paramters (on-time-ratio rho, brightness eps)
    rho_map, rho_tru, eps_map = esti_rhoeps(xn_set, res)

    return k_set, rho_map, eps_map