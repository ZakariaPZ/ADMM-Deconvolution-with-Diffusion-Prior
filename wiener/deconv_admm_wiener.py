import numpy as np
from tqdm import tqdm
from numpy.fft import fft2, ifft2
from pypher.pypher import psf2otf
from wiener.wiener_prior import wiener

# PyTorch
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def deconv_admm_wiener(b, c, lam, rho, num_iters, noise_sigma):

    # Blur kernel
    cFT = psf2otf(c, b.shape)
    cTFT = np.conj(cFT)

    # Fourier transform of b
    bFT = fft2(b)

    # initialize x,z,u with all zeros
    x = np.zeros_like(b)
    z = np.zeros_like(b)
    u = np.zeros_like(b)

    denom = (cTFT * cFT) + rho  # you need to edit this placeholder

    for it in tqdm(range(num_iters)):

        v = z - u

        x = ifft2((cTFT * bFT + (rho * fft2(v)))/denom)

        # z update
        v = x + u
        z = wiener(v, cFT, noise_sigma)

        # u update
        u = u + x - z

    return x