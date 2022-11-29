import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift

def wiener(blurred_img, cFT, noise_sigma):
    cFT_shift = fftshift(cFT)
    blurred_img_FT = fftshift(fft2(blurred_img))

    unblur_kernel = (1/cFT_shift) * (np.power(np.abs(cFT_shift), 2)/(np.power(np.abs(cFT_shift), 2) + (noise_sigma/np.average(blurred_img))))
    return np.real(ifft2(ifftshift(np.multiply(blurred_img_FT, unblur_kernel))))