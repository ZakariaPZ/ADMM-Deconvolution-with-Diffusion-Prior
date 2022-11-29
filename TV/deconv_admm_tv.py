import numpy as np
from numpy.fft import fft2, ifft2
from pypher.pypher import psf2otf
from pathlib import Path
from tqdm import tqdm
from itertools import product
import matplotlib.pyplot as plt
import torch


def deconv_admm_tv(b, c, lam, rho, num_iters, anisotropic_tv=False):

    # Blur kernel
    cFT = psf2otf(c, b.shape)
    cTFT = np.conj(cFT)

    # First differences
    dx = np.array([[-1., 1.]])
    dy = np.array([[-1.], [1.]])
    dxFT = psf2otf(dx, b.shape)
    dyFT = psf2otf(dy, b.shape)
    dxTFT = np.conj(dxFT)
    dyTFT = np.conj(dyFT)
    dxyFT = np.stack((dxFT, dyFT), axis=0)
    dxyTFT = np.stack((dxTFT, dyTFT), axis=0)

    # Fourier transform of b 
    bFT = fft2(b)

    # initialize x,z,u with all zeros
    x = np.zeros_like(b)
    z = np.zeros((2, *b.shape))
    u = np.zeros((2, *b.shape))

    ################# begin task 2 ###################################
    # Complete these parts by first copying over the grad_fn you
    # implemented for task 1 here. What's important here (and wasn't
    # in task 1) is that grad_fn takes as input a 2D image of size [M N]
    # and outputs the horizontal and vertical gradients in a stack of
    # size [2 M N]! Please keep this in mind, otherwise the z-update
    # which is already implemented for you won't work

    # Then, you can pre-compute the denominator for the x-update
    # here, because that doesn't change unless rho changes, which is
    # not the case here

    # define function handle to compute horizontal and vertical gradients
    grad_fn = lambda x: np.real(ifft2(fft2(x) * dxyFT))   # you need to edit this, it's just a placeholder

    # precompute the denominator for the x-update
    denom = (cTFT * cFT) + rho*(dxyTFT[0] * dxyFT[0] + dxyTFT[1] * dxyFT[1])   # you need to edit this, it's just a placeholder

    ################# end task 2 ####################################

    for it in tqdm(range(num_iters)):

        ################# begin task 2 ###################################

        # Complete this part by implementing the x-update discussed in
        # class and in the problem session. If you implemented the
        # denominator term above, you only need to compute the nominator
        # here as well as the rest of the x-update

        # x update - inverse filtering: Fourier multiplications and divisions
        v = z - u
        vFT = fft2(v)
        x = ifft2((cTFT * bFT + rho * (dxyTFT[0] * vFT[0] + dxyTFT[1] * vFT[1]))/denom)

        ################# end task 2 ####################################

        # z update - soft shrinkage
        kappa = lam / rho
        v = grad_fn(x) + u

        # proximal operator of anisotropic TV term
        if anisotropic_tv:
            z = np.maximum(1 - kappa/np.abs(v), 0) * v

        # proximal operator of isotropic TV term
        else:
            vnorm = np.sqrt( v[0,:,:]**2 + v[1,:,:]**2 )
            z[0,:,:] = np.maximum(1 - kappa/vnorm,0) * v[0,:,:]
            z[1,:,:] = np.maximum(1 - kappa/vnorm,0) * v[1,:,:]

        # u-update
        u = u + grad_fn(x) - z

    return x
