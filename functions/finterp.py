import numpy as np
import imageio
from numpy.fft import (fftshift, ifftshift, fft, ifft)
import tifffile as tiff 

# Set up base vectors.
def base_vect_generator2D(xrange, yrange):
    bx, by = np.zeros(xrange), np.zeros(yrange)
    bx[1], by[1] = 1, 1
    bx, by = np.fft.fft(bx), np.fft.fft(by)
    # bx, by = fftshift(fft(ifftshift(bx))), fftshift(fft(ifftshift(by)))
    return bx, by

# Define Fourier transform metrix.
def calc_ft_matrix(base, spectrum_range):
    power_matrix = np.ones((spectrum_range, spectrum_range))
    power_matrix = np.arange(spectrum_range).reshape(spectrum_range, 1) * power_matrix
    ft_matrix = np.power(base, power_matrix)
    return ft_matrix
    
def ft_matrix2D(xrange, yrange):
    bx, by = base_vect_generator2D(xrange, yrange)
    fx = calc_ft_matrix(bx, xrange)
    fy = calc_ft_matrix(by, yrange).T
    return fx, fy

# Define inverse Fourier transform metrix.
def calc_ift_matrix(base, spectrum_range, interp_num):
    conj_base = np.reshape(np.conj(base), (1, spectrum_range))
    ift = np.matmul(np.ones(((spectrum_range - 1) * interp_num + 1, 1)),conj_base)
    iftp = np.arange(0, spectrum_range - 1 + 1e-10, 1/interp_num)
    iftp = np.matmul(iftp.reshape(-1, 1), np.ones((1, spectrum_range)))
    ift = np.power(ift, iftp) / spectrum_range
    return ift
    
def ift_matrix2D(xrange, yrange, interp_num):
    '''
    xrange: int, 2 times the dimension x of a frame
    yrange: int, 2 times the dimension y of a frame
    interp_num: int, the number of times the resolution enhanced, 
        e.g. when interp_num = 2, d' = d / 2, the resolution if two times the original one.
    '''
    bx, by = base_vect_generator2D(xrange, yrange)
    ifx = calc_ift_matrix(bx, xrange, interp_num)
    ify = calc_ift_matrix(by, yrange, interp_num).T
    
    return ifx, ify

def interpolate_image(im, fx, fy, ifx, ify, xdim, ydim, interp_num):
    # [im,fliplr(im);flipud(im),rot90(im,2)] is mirror-extension of the image A to create the natural peoriocity in the resulting image to avoid ringing artifacts after fourier interpolation.
    ext_im = np.append(np.append(im, np.fliplr(im), axis=1), 
                       np.append(np.flipud(im), np.rot90(im, 2), axis=1), axis=0)

    # Fourier transform
    fall = np.matmul(np.matmul(fx, ext_im), fy)     # fall=fx@ext_im@fy for Python 3.5 or newer versions

    # Inverse Fourier transform
    ifall = np.absolute(np.dot(np.dot(ifx, fall),ify))

    # Take the region corresponding to the FOV of the original image
    xdim_new = (xdim - 1) * interp_num + 1
    ydim_new = (ydim - 1) * interp_num + 1
    interp_im = ifall[:xdim_new, :ydim_new]
    
    return interp_im

def fourier_interp_tiffimage(im, interp_num_lst):
    # get dimensions and number of frames
    xdim, ydim = im.shape
    xrange, yrange = 2 * xdim, 2 * ydim    # define the ft spectrum span
    fx, fy = ft_matrix2D(xrange, yrange)
    interp_im_lst = []
    for interp_num in interp_num_lst:
        ifx, ify = ift_matrix2D(xrange, yrange, interp_num)
        interp_im = interpolate_image(im, fx, fy, ifx, ify, xdim, ydim, interp_num)       
        interp_im_lst.append(np.int_(np.around(interp_im)))
            
    return interp_im_lst

def fourier_interp_tiffstack(filepath, filename, interp_num_lst, mvlength = None, save_option = True, return_option = False):

    imstack = tiff.TiffFile(filepath + '/' + filename + '.tif')
    xdim, ydim = np.shape(imstack.pages[0])
    xrange, yrange = 2 * xdim, 2 * ydim
    fx, fy = ft_matrix2D(xrange, yrange)
    interp_imstack_lst = []
    
    # if user did not select the video length, process on the whole video
    if mvlength == None:
        mvlength = len(imstack.pages)
        
    for interp_num in interp_num_lst:
        ifx, ify = ift_matrix2D(xrange, yrange, interp_num)
        interp_imstack = []

        for frame in range(mvlength):
            im = tiff.imread(filepath + '/' + filename + '.tif', key=frame)
            interp_im = interpolate_image(im, fx, fy, ifx, ify, xdim, ydim, interp_num)
            interp_im = np.int_(np.around(interp_im))
            if save_option == True:
                tiff.imwrite(filename + '_InterpNum' + str(interp_num) + '.tif', interp_im, dtype='int', append=True)
            if return_option == True:
                interp_imstack.append(interp_im)
                
        if return_option == True:  
            interp_imstack_lst.append(interp_imstack) 
    
    if return_option == True:
        return interp_imstack_lst
