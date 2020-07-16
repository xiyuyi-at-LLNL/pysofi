import numpy as np
from scipy.interpolate import griddata

def ldrc(input_im, mask_im, order, window_size):
    xdim_mask, ydim_mask = np.shape(mask_im)
    xdim, ydim = np.shape(input_im)
    if xdim == xdim_mask and ydim == ydim_mask:
        mask = mask_im
    else:
        # resize mask to the image dimension if not the same dimension (xdim = xdim_mask, ydim = ydim_mask)
        mod_xdim = (xdim_mask-1)*order + 1    # new mask x dimemsion
        mod_ydim = (ydim_mask-1)*order + 1    # new mask y dimemsion
        px = np.arange(0,mod_xdim,order)
        py = np.arange(0,mod_ydim,order)
    
        # create coordinate list for interpolation
        coor_lst = [] 
        for i in px:
            for j in py:
                coor_lst.append([i,j])
        coor_lst = np.array(coor_lst)

        orderjx = complex(str(mod_xdim) + 'j')
        orderjy = complex(str(mod_ydim) + 'j')
        px_new, py_new = np.mgrid[0:mod_xdim-1:orderjx, 0:mod_ydim-1:orderjy]    # new coordinates for interpolated mask

        interp_mask = griddata(coor_lst, mask_im.reshape(-1,1), (px_new, py_new), method='cubic')    # interpolation
        mask = interp_mask.reshape(px_new.shape)
    
    # input_im, mask, window_size, xdim, ydim
    seq_map = np.zeros((xdim, ydim))
    ldrc_im = np.zeros((xdim, ydim))
    for i in range(xdim - window_size + 1):
        for j in range(ydim - window_size + 1):
            window = input_im[i:i+window_size, j:j+window_size]
            norm_window = (window - np.min(window)) / (np.max(window) - np.min(window))
            # norm_window = window / np.max(window)
            ldrc_im[i:i+window_size, j:j+window_size] = ldrc_im[i:i+window_size, j:j+window_size] + norm_window * np.max(mask[i:i+window_size, j:j+window_size])
            seq_map[i:i+window_size, j:j+window_size] = seq_map[i:i+window_size, j:j+window_size] + 1
        
    ldrc_im = ldrc_im / seq_map  
    return ldrc_im

