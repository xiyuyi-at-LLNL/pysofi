import numpy as np
import imageio
import scipy.special

def average_image(filepath, filename):
    '''
    Get the average image for a video file (tiff stack).
    '''
    # TODO: the user can define the number of sub-blocks, and get average image for each bloack.
    im = imageio.imread(filepath + '/' + filename)
    xdim, ydim = np.shape(im)    # get dimensions
    mean_im = np.zeros((im.shape))
    
    vid_reader = imageio.get_reader(filepath + '/' + filename)
    mvlength = vid_reader.get_length()
    for frame in vid_reader:
        mean_im = mean_im + frame    
    mean_im = mean_im / mvlength
    
    return mean_im

def calc_moments(filepath, filename, highest_order):
    '''
    Get all moment-reconstructed images to the user-defined highest order for a video (tiff stack).
    '''
    # TODO: the user can define the number of sub-blocks, and get moment images for each bloack.
    mean_im = average_image(filepath, filename)
    xdim, ydim = np.shape(mean_im)
    m_set = {}
    calculated_moment_images = []
    vid_reader = imageio.get_reader(filepath + '/' + filename)
    mvlength = vid_reader.get_length()
    
    for order in range(highest_order):
        m_set[order+1] = np.zeros((xdim, ydim))
        for frame in vid_reader:
            m_set[order+1] = m_set[order+1] + np.power(frame - mean_im, order+1)
        
        m_set[order+1] = np.int64(m_set[order+1] / mvlength)
    return m_set

def calc_cumulants_from_moments(moment_set, highest_order):
    '''
    Calculate cumulant-reconstructed images from moment-reconstructed images.
    '''
    k_set = {}
    for order in range(1, highest_order + 1):
        k_set[order] = np.int64(moment_set[order] - np.sum(np.array([scipy.special.comb(order-1,i)*k_set[order-i]*moment_set[i] for i in range(1,order)]),axis=0))
    
    return k_set


