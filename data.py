from functions import (deconvsk, finterp, ldrc, masks, reconstruction)
import tifffile as tiff
import numpy as np

class Data:
    def __init__(self, filepath, filename):
        self.filename = filename
        self.filepath = filepath
        self.ave = None
        self.moments_set = None
        self.cumulants_set = None
        self.morder = 0
        self.corder = 0
        
    def average_image(self):
        self.ave = reconstruction.average_image(self.filepath, self.filename)
        return self.ave
    
    def moments_images(self, highest_order = 6):
        self.morder = highest_order
        self.moments_set = reconstruction.calc_moments(self.filepath, self.filename, highest_order)
        return self.moments_set
    
    def cumulants_images(self, highest_order = 6, m_set = None, same_order = True):
        self.corder = highest_order
        if m_set is None:
            if self.moments_set is not None and self.morder >= self.corder: 
                m_set = self.moments_set
            else:
                m_set = self.moments_images(highest_order)
              
        if same_order is False:
            m_set = self.moments_images(highest_order)
            
        self.cumulants_set = reconstruction.calc_cumulants_from_moments(m_set, highest_order)
        return self.cumulants_set
            
    def ldrc(self, order = 6, window_size = 25, mask_im = None, input_im = None):      
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
    
    def deconvsk(self, est_psf = masks.gauss2D_mask((51, 51), 2), input_im = None, deconv_lambda = 1.5, deconv_iter = 20):
        if input_im is None:
            if self.moments_set is not None:
                input_im = self.moments_set[self.morder]
            elif self.ave is not None:
                input_im = self.ave
            else:
                input_im = self.average_image()
        self.deconv = deconvsk.deconvsk(est_psf, input_im, deconv_lambda, deconv_iter)
        return self.deconv
    
    def finterp_tiffstack(self, interp_num_lst = [2,4], mvlength = None, save_option = True, return_option = False):
        filename = self.filename[:-4]
        if return_option == False:
            finterp.fourier_interp_tiffstack(self.filepath, filename, interp_num_lst, mvlength, save_option, return_option)
        else:
            self.finterps = finterp.fourier_interp_tiffstack(self.filepath, filename, interp_num_lst, mvlength, save_option, return_option)
            return self.finterps
       
    def finterp_image(self, input_im = None, interp_num_lst = [2,4]):
        if input_im is None:
            input_im = self.average_image()
            
        self.finterp = finterp.fourier_interp_tiffimage(input_im, interp_num_lst)
        return self.finterp
        
    def add_filtered(self, image, filter_name='noise filter'):
        self.filter_name = filter_name
        self.filtered = image        

    def get_length(self):
        imstack = tiff.TiffFile(filepath + '/' + filename)
        self.n_frames = len(imstack.pages)
        return self.n_frames