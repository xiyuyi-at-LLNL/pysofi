from . import (deconvsk, finterp, ldrc, masks, reconstruction)
import tifffile as tiff
import numpy as np

class Data:
    '''
    Data object contains the information of a dataset (e.g. dimensions,
    frame numbers), and provides SOFI methods to perform analysis and 
    visualization on the dataset (moments reconstruction, cumulants 
    reconstruction, shrinking kernel deconvolution, etc...). 
    
    When loading a tiff file, a Data() object is created and further 
    SOFI analysis can be preformed. All the standard data-attributes 
    are listed below. New data-attributes can be added or updated using 
    '.add()' function.

    Parameters
    ----------
    filapath: str
        Path to the tiff file.
    filename: str
        Name of the tiff file.

    Attributes
    ----------
    filename: str
    filepath: str
    ave: ndarray
        The average image of the image stack / tiff video.
    moments_set: dict
        moment order (int) -> moment-reconstructed image (ndarray)
        A dictionary of orders and corrensponding reconstructions.
    cumulants_set: dict
        cumulant order (int) -> cumulant-reconstructed image (ndarray)
        A dictionary of orders and corrensponding reconstructions.    
    morder: int
        The highest order of moment reconstruction that has been calculated.
    corder: int
        The highest order of cumulant reconstruction that has been calculated.    
    n_frames: int
        The number of frames of the image stack / tiff video.
    xdim: int
        The number of pixels in x dimension.
    ydim: int
        The number of pixels in y dimension.


    Notes
    -----
    For SOFI processing, after loading the tiff video into the Data object, 
    a pipeline of 1) fourier interpolation, 2) moments reconstrtuction or 
    cumulants reconstruction, 3) noise filtering 1, 4) shrinking kernel de-
    convolution, 5) noise filtering 2, and 6) ldrc. The processed video will
    be saved into a new tiff file with the colormap user selects.

    References
    ----------
    .. [1] Xiyu Yi, Sungho Son, Ryoko Ando, Atsushi Miyawaki, and Shimon Weiss, 
    "Moments reconstruction and local dynamic range compression of high order 
    superresolution optical fluctuation imaging," Biomed. Opt. Express 10, 
    2430-2445 (2019).

    '''
    def __init__(self, filepath, filename):
        self.filename = filename
        self.filepath = filepath
        self.ave = None
        self.moments_set = {}
        self.cumulants_set = {}
        self.morder = 0
        self.corder = 0
        self.n_frames, self.xdim, self.ydim = self.get_dims()
        
    def average_image(self):
        '''Calculate average image of the tiff video.'''
        self.ave = reconstruction.average_image(self.filepath, self.filename)
        return self.ave
    
    def moments_images(self, highest_order = 6, mean_im = None):
        '''
        Calculate moment-reconstructed images to the highest order. 
        Parameters
        ----------
        highest_order: int
            The highest order number of moment-reconstructed images.
        mean_im: ndarray
            Average image of the tiff stack.

        Returns
        -------
        moments_set: dict
            order number (int) -> image (ndarray)
            A dictionary of calcualted moment-reconstructed images.
        '''
        if mean_im is None and self.ave is not None:
            mean_im = self.ave
        self.moments_set = reconstruction.calc_moments(self.filepath, 
            self.filename, highest_order, self.moments_set, mean_im)
        self.morder = highest_order
        return self.moments_set
    
    def cumulants_images(self, highest_order=6, m_set=None):
        '''
        Calculate cumulant-reconstructed images to the highest order. 
        Parameters
        ----------
        highest_order: int
            The highest order number of cumulant-reconstructed images.
        m_set: dict
            order number (int) -> image (ndarray)
            A dictionary of calcualted moment-reconstructed images.

        Returns
        -------
        cumulants: dict
            order number (int) -> image (ndarray)
            A dictionary of calcualted cumulant-reconstructed images.
        '''
        self.corder = highest_order
        if m_set is None:    # moments not provided
            if self.moments_set == {}:    # moments have not calculated
                m_set = self.moments_images(highest_order)
            else:
                if self.corder > self.morder:
                    m_set = self.moments_images(highest_order)
                else:
                    m_set = self.moments_set
        else:
            if self.corder > self.morder:
                m_set = self.moments_images(highest_order)
            else:
                m_set = self.moments_set
            
        self.cumulants_set = reconstruction.calc_cumulants_from_moments(m_set)
        return self.cumulants_set
    
            
    def ldrc(self, order=6, window_size=[25,25], mask_im=None, input_im=None): 
        '''
        Compress the dynamic range of moments-/cumulants reconstruced image 
        with ldrc method. For details of ldrc, refer to ldrc.py and [1].

        Parameters
        ----------
        order: int
            The order of the reconstructed image.
        window_size: [int, int]
            The [x, y] dimension of the scanning window.
        mask_im: ndarray
            A reference image. 
            Usually a average/sum image or second-order SOFI image is used.
        input_im: ndarray
            An input image, usually a high-order moment- or cumulant- 
            reconstructed image.

        Returns
        -------
        ldrc_image: ndarray
            The compressed image with the same dimensions of input_im.
        '''
        if input_im is None: 
            if self.moments_set is not None:
                if self.morder >= order:
                    input_im = self.moments_set[order]
                else:
                    self.moments_images(order)[order]
            else:
                self.moments_images(order)[order]
                
        if mask_im is None:
                mask_im = self.average_image()                

        # input_im = self.moments_set[4], mask_im = self.ave    
        self.ldrc_image = ldrc.ldrc(mask_im, input_im, order, window_size)
        return self.ldrc_image
    
    def deconvsk(self, est_psf=masks.gauss2D_mask((51,51),2), input_im=None,
                 deconv_lambda=1.5, deconv_iter=20):
        '''
        Perform serial Richardson-Lucy deconvolution with a series of PSFs
        on the input image. For details of shrinking kernel deconvolution,
        refer to deconvsk.py and [1].

        Parameters
        ----------
        est_psf: ndarray
            Estimated PSF.
        input_im: ndarray
            Input image that need deconvolution.
        deconv_lambda: float
            Lambda for the exponent between. It is an empirical parameter
            within the range of (1,2).
        deconv_iter: int
            Number of iterations for each deconvolution.

        Returns
        -------
        deconv: ndarray
            Deconvoluted image.
        '''
        if input_im is None:
            if self.moments_set is not None:
                input_im = self.moments_set[self.morder]
            elif self.ave is not None:
                input_im = self.ave
            else:
                input_im = self.average_image()
        self.deconv = deconvsk.deconvsk(est_psf, input_im, 
                                        deconv_lambda, deconv_iter)
        return self.deconv
    
    def finterp_tiffstack(self, interp_num_lst = [2,4], mvlength = None, 
                          save_option = True, return_option = False):
        '''
        Performs fourier interpolation on a tiff image (stack) with a list of 
        interpolation factors (the number of pixels to be interpolated between 
        two adjacent pixels).

        Parameters
        ----------
        interp_num_lst: list (int)
            A list of interpolation factors.
        mvlength: int
            The frame number of the tiff file (1 for images).
        save_option: bool
            Whether to save the interpolated images into tiff files (each 
            interpolation factor seperately).
        return_option: bool
            Whether to return the interpolated image series as 3d arrays.

        Returns
        -------
        finterps: list (ndarray)
            A list of interpolated image sereis corresponding to the interpolation 
            factor list.
        '''
        filename = self.filename[:-4]
        if return_option == False:
            finterp.fourier_interp_tiff(self.filepath, filename, 
                                        interp_num_lst, mvlength, 
                                        save_option, return_option)
        else:
            self.finterps = finterp.fourier_interp_tiff(self.filepath, 
                filename, interp_num_lst, mvlength, save_option, return_option)
            return self.finterps
       
    def finterp_image(self, input_im = None, interp_num_lst = [2,4]):
        '''Performs fourier interpolation on an image array.'''
        if input_im is None:
            input_im = self.average_image()
            
        self.finterp = finterp.fourier_interp_array(input_im, interp_num_lst)
        return self.finterp
        
    def add_filtered(self, image, filter_name='noise filter'):
        '''Add (noise) filtered image as an attribute of the object.'''
        self.filter_name = filter_name
        self.filtered = image        

    def get_dims(self):
        '''Get dimensions and frame number of the tiff video.'''
        imstack = tiff.TiffFile(self.filepath + '/' + self.filename)
        n_frames = len(imstack.pages)
        xdim, ydim = np.shape(imstack.pages[0])
        return n_frames, xdim, ydim

    def add(self, **kwargs):
        """Adds or updates elements (attributes and/or dict entries). """
        for k, v in kwargs.items():
            setattr(self, k, v)

