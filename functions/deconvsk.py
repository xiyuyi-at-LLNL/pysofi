from numpy.fft import (fftshift, ifftshift, fftn, ifftn, rfftn, irfftn)
import numpy as np

def _prep_img_and_psf(image, psf):
    """Do basic data checking, convert data to float, normalize psf and make
    sure data are positive"""
    #assert psf.ndim == image.ndim, ("image and psf do not have the same number"
     #                               " of dimensions")
    image = image.astype(np.float)
    psf = psf.astype(np.float)
    # need to make sure both image and PSF are totally positive.
    image = _ensure_positive(image)
    psf = _ensure_positive(psf)
    # normalize the kernel
    psf /= psf.sum()
    return image, psf

def _ensure_positive(data):
    """Make sure data is positive and has no zeros

    For numerical stability

    If we realize that mutating data is not a problem
    and that changing in place could lead to signifcant
    speed ups we can lose the data.copy() line"""
    # make a copy of the data
    data = data.copy()
    data[data <= 0] = np.finfo(data.dtype).eps
    return data

def _zero2eps(data):
    """Make sure data is positive and has no zeros

    For numerical stability

    If we realize that mutating data is not a problem
    and that changing in place could lead to signifcant
    speed ups we can lose the data.copy() line"""
    # make a copy of the data
    return np.fmax(data, np.finfo(data.dtype).eps)

# https://github.com/aboucaud/pypher/blob/master/pypher/pypher.py
def zero_pad(image, shape, position='center'):
    """
    Extends image to a certain size with zeros
    Parameters
    ----------
    image: real 2d `numpy.ndarray`
        Input image
    shape: tuple of int
        Desired output shape of the image
    position : str, optional
        The position of the input image in the output one:
            * 'corner'
                top-left corner (default)
            * 'center'
                centered
    Returns
    -------
    padded_img: real `numpy.ndarray`
        The zero-padded image
    """
    shape = np.asarray(shape, dtype=int)
    imshape = np.asarray(image.shape, dtype=int)

    if np.alltrue(imshape == shape):
        return image

    if np.any(shape <= 0):
        raise ValueError("ZERO_PAD: null or negative shape given")

    dshape = shape - imshape
    if np.any(dshape < 0):
        raise ValueError("ZERO_PAD: target size smaller than source one")

    pad_img = np.zeros(shape, dtype=image.dtype)

    idx, idy = np.indices(imshape)

    if position == 'center':
        if np.any(dshape % 2 != 0):
            raise ValueError("ZERO_PAD: source and target shapes "
                             "have different parity.")
        offx, offy = dshape // 2
    else:
        offx, offy = (0, 0)

    pad_img[idx + offx, idy + offy] = image

    return pad_img


def psf2otf(psf, shape):
    """
    Convert point-spread function to optical transfer function.
    Compute the Fast Fourier Transform (FFT) of the point-spread
    function (PSF) array and creates the optical transfer function (OTF)
    array that is not influenced by the PSF off-centering.
    By default, the OTF array is the same size as the PSF array.
    To ensure that the OTF is not altered due to PSF off-centering, PSF2OTF
    post-pads the PSF array (down or to the right) with zeros to match
    dimensions specified in OUTSIZE, then circularly shifts the values of
    the PSF array up (or to the left) until the central pixel reaches (1,1)
    position.
    Parameters
    ----------
    psf : `numpy.ndarray`
        PSF array
    shape : int
        Output shape of the OTF array
    Returns
    -------
    otf : `numpy.ndarray`
        OTF array
    Notes
    -----
    Adapted from MATLAB psf2otf function
    """
    if np.all(psf == 0):
        return np.zeros_like(psf)

    inshape = psf.shape
    # Pad the PSF to outsize
    psf = zero_pad(psf, shape, position='corner')

    # Circularly shift OTF so that the 'center' of the PSF is
    # [0,0] element of the array
    for axis, axis_size in enumerate(inshape):
        psf = np.roll(psf, -int(axis_size / 2), axis=axis)

    # Compute the OTF
    otf = np.fft.fft2(psf)

    # Estimate the rough number of operations involved in the FFT
    # and discard the PSF imaginary part if within roundoff error
    # roundoff error  = machine epsilon = sys.float_info.epsilon
    # or np.finfo().eps
    n_ops = np.sum(psf.size * np.log2(psf.shape))
    otf = np.real_if_close(otf, tol=n_ops)

    return otf

def otf2psf(otf, shape):

    if np.all(otf == 0):
        return np.zeros_like(otf)

    inshape = otf.shape
    
    # Compute the PSF
    psf = np.fft.ifft2(otf)
    
    # Estimate the rough number of operations involved in the FFT
    # and discard the PSF imaginary part if within roundoff error
    # roundoff error  = machine epsilon = sys.float_info.epsilon
    # or np.finfo().eps
    n_ops = np.sum(otf.size * np.log2(otf.shape))
    psf = np.real_if_close(psf, tol=n_ops)
    psf=np.real(psf)
    
    
    # Circularly shift PSF so that the 'center' of the PSF is
    # [0,0] element of the array
    for axis, axis_size in enumerate(shape):
        psf = np.roll(psf, int(axis_size / 2), axis=axis)
    
    # Crop output array
    psf = psf[0:shape[0], 0:shape[1]]

    return psf

def corelucy(image, H):
    u_t = image
    reblur = np.real(ifftn(H * fftn(u_t, u_t.shape), u_t.shape))
    reblur = _ensure_positive(reblur)
    im_ratio = image / reblur
    f = fftn(im_ratio)
    return f

# image, psf = _prep_img_and_psf(ext_im, deconv_psf0)

def richardson_lucy(image, psf, iterations=10, **kwargs):
    
    # 1. prepare parameters
    image, psf = _prep_img_and_psf(image, psf)
    sizeI, sizePSF = image.shape, psf.shape
    J, P = {}, {}
    J[1], J[2], J[3], J[4] = image, image, 0, np.zeros((np.prod(sizeI), 2))
    P[1], P[2], P[3], P[4] = psf, psf, 0, np.zeros((np.prod(sizePSF), 2))
    WEIGHT = np.ones(image.shape)
    fw = fftn(WEIGHT)
    
    # 2. L_R iterations
    for k in range(iterations):
        # 2a. make image and PSF predictions for the next iteration
        Y = np.maximum(J[2],0)
        B = np.maximum(P[2],0)
        B /= B.sum()
        # 2b. make core for the LR estimation
        H = psf2otf(B, sizeI) 
        CC = corelucy(Y, H)
        # 2c. Determine next iteration image & apply positivity constraint
        J[3] = J[2]
        scale = np.real(ifftn(np.multiply(np.conj(H),fw))) + np.sqrt(np.finfo(H.dtype).eps)
        J[2] = np.maximum(np.multiply(image, np.real(ifftn(np.multiply(np.conj(H), CC))))/scale, 0)
        J[4] = np.vstack([J[2].T.reshape(-1,) - Y.T.reshape(-1,), J[4][:,1]]).T
        # 2d. Determine next iteration PSF & apply positivity constraint + normalization
        P[3] = P[2]
        H = fftn(J[3])
        scale = otf2psf(np.multiply(np.conj(H),fw), sizePSF) + np.sqrt(np.finfo(H.dtype).eps)
        P[2] = np.maximum(np.multiply(B, otf2psf(np.multiply(np.conj(H),CC), sizePSF))/scale, 0)
        P[2] /= P[2].sum()
        P[4] = np.vstack([P[2].T.reshape(-1,) - B.T.reshape(-1,), P[4][:,1]]).T
    P, J = P[2], J[2]  # PSF and updated image
    return P, J  

def deconvsk(est_psf, input_im, deconv_lambda, deconv_iter):
    xdim, ydim = np.shape(input_im)
    deconv_im = np.append(np.append(input_im, np.fliplr(input_im), axis=1), np.append(np.flipud(input_im), np.rot90(input_im, 2), axis=1), axis=0)
    # perform mirror extension to the image in order to surpress ringing artifacts associated with fourier transform due to truncation effect.
    psf0 = est_psf / np.max(est_psf)
    for iter_num in range(deconv_iter):
        alpha = deconv_lambda**(iter_num+1) / (deconv_lambda - 1)
        deconv_psf, deconv_im = richardson_lucy(deconv_im, psf0**alpha, 1)    
    
    deconv_im = deconv_im[0:xdim, 0:ydim]
    return deconv_im