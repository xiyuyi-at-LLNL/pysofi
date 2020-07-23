import numpy as np
from . import masks

def filter1D_same(time_series, noise_filter):
    '''
    Filter original time_series with noise_filter, and return the filtered_series with the same 
    length as the original sereis.
    Compared to MATLAB results, when the length of the filter and frame number are both even, 
    the filtered result would shift to left by one number. In other cases, results are the same.
    '''
    filtered_series = np.convolve(time_series, noise_filter, 'full')
    filtered_center = len(filtered_series)//2
    original_center = len(time_series)/2
    filtered_series = filtered_series[np.int(filtered_center - np.floor(original_center)):np.int(filtered_center + np.ceil(original_center))]
    return filtered_series

def noise_filter(dset, im_set, noise_filter = masks.gauss1D_mask((1,21), 2), filtername = 'noise filter after M6', filenames = None, return_option = False):
    if filenames is None:
        filenames =[*im_set]
    get_series = lambda i,j: [im_set[filename][i,j] for filename in filenames]
    xdim, ydim = np.shape(im_set[filenames[0]])
    m_length = len(filenames)
    m_filtered = np.zeros((m_length, xdim, ydim))
    for i in range(xdim):
        for j in range(ydim):
            time_series = get_series(i,j)
            m_filtered[:,i,j] = filter1D_same(time_series, noise_filter)
    for k in range(m_length):
        dset[filenames[k]].add_filtered(m_filtered[k], filtername)
        
    if return_option == True:
        return m_filtered
