import numpy as np
from tqdm import tqdm
from numpy.fft import fft2, ifft2
from pypher.pypher import psf2otf

# PyTorch
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from bilateral.bilateral import bilateral2d


# from network_dncnn import DnCNN as net


def deconv_admm_bilateral(b, c, lam, rho, num_iters, sigma, sigmaIntensity):

    # Blur kernel
    cFT = psf2otf(c, b.shape)
    cTFT = np.conj(cFT)

    # Fourier transform of b
    bFT = fft2(b)

    # initialize x,z,u with all zeros
    x = np.zeros_like(b)
    z = np.zeros_like(b)
    u = np.zeros_like(b)

    # print(x.shape)

    # set up NLM model
    averageFilterRadius = int(sigma)

    ################# begin task 2 ###################################

    # ADMM with DnCNN doesn't require a gradient function, so we don't
    # need it here. Just pre-compute the denominator for the x-update
    # here, because that doesn't change unless rho changes, which is
    # not the case here

    # pre-compute denominator of x update
    denom = cTFT * cFT + rho  # you need to edit this placeholder

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
        x = np.real(ifft2((cTFT * bFT + rho*vFT)/denom))

        ################# end task 2 ####################################

        # z update
        v = x + u

        # run DnCNN denoiser
        # v_tensor = torch.reshape(torch.from_numpy(v).float().to(device), (1, 1, v.shape[0], v.shape[1]))

        v_tensor_denoised = bilateral2d(v,
                                            sigma,
                                            sigmaIntensity)

        z = np.squeeze(v_tensor_denoised)

        # u update
        u = u + x - z

    return x
