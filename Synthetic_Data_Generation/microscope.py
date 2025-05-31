import numpy as np
from scipy import special
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
from skimage.util import random_noise
from skimage.transform import resize
from utils import normalize_image


class Microscope:
    def __init__(self, p_real, Lambda, NA, n):
        self.p_real = p_real              #scale, in nm/px
        self.Lambda = Lambda    #emission wavelength
        self.NA = NA            #numerical aperture
        self.n = n              #refractive index
        
    def get_PSF(self, p_gen):  #credit to Georgeos
        halfsize = 6*self.Lambda/(2*np.pi*self.NA*p_gen)
        r = np.arange(-halfsize,halfsize+1) 
        kem = 2*np.pi/self.Lambda
        #kaw = self.NA/self.n * 2 * np.pi/self.Lambda
        xx,yy = np.meshgrid(r,r) 
        rr = np.sqrt(xx**2+yy**2) * kem * self.NA * p_gen
        PSF = (2*special.jv(1,rr)/(rr))**2
        self.PSF = PSF
        
        # plt.figure(figsize=(4.5,4.5))
        # plt.imshow(np.sqrt(PSF),cmap="Greys_r")
        # scalebar = ScaleBar(p_gen, 'nm')
        # plt.gca().add_artist(scalebar)
        # plt.show()
        
        return PSF

    def convolve_with_Gaussian(self, image, p_gen): #input image must be within range [0.0,1.0]
        gauss_sigma = 0.3*self.Lambda/self.NA #0.21
        convolved_image = gaussian_filter(image, sigma=gauss_sigma/p_gen, mode='constant')
        convolved_image[convolved_image>1.0] = 1.0
        #convolved_image = (65535*convolved_image).astype('uint16')
        return convolved_image

    def convolve_with_PSF(self, image): #input image must be within range [0.0,1.0]
        convolved_image = convolve2d(image, self.PSF, "same")
        convolved_image = normalize_image(convolved_image)
        #convolved_image = (65535*convolved_image).astype('uint16')
        return convolved_image

    def resize_image(self, image, new_size):
        image_resized = resize(image, new_size, anti_aliasing=True)
        return image_resized

    def add_noise(self, image, mode_name, sigma):
        if mode_name=='gaussian':
            noisy = random_noise(image, mode='gaussian', var=sigma**2)
            return noisy
        elif mode_name=='poisson':
            noisy = random_noise(image, mode='poisson')
            return noisy
        else:
            print("Unsupported noise type")
            return None
        #if renormalize:
        #    noisy = normalize_image(noisy)
        