import numpy as np
from tqdm import tqdm
from numpy.fft import fft2, ifft2
from pypher.pypher import psf2otf

# PyTorch
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def deconv_admm_dncnn(b, c, lam, rho, num_iters, model):

    # initialize x,z,u with all zeros
    x = np.zeros_like(b)
    z = np.zeros_like(b)
    u = np.zeros_like(b)
    v = np.zeros_like(b)

    # set up DnCNN model
    model = model

    model.eval()
    for k, l in model.named_parameters():
        l.requires_grad = False
    model = model.to(device)

    num_channels = 3

    ##### Begin ADMM

    for it in tqdm(range(num_iters)):

        for channel in range(num_channels):

            # Blur kernel
            cFT = psf2otf(c, b[:, :, channel].shape)
            cTFT = np.conj(cFT)

            # Fourier transform of b
            bFT = fft2(b[:, :, channel])

            # pre-compute denominator of x update
            denom = cTFT * cFT + rho  

            # x update - inverse filtering: Fourier multiplications and divisions

            v[:, :, channel] = z[:, :, channel] - u[:, :, channel]
            vFT = fft2(v[:, :, channel])
            x[:, :, channel]  = np.real(ifft2((cTFT * bFT + rho*vFT)/denom))


        # z update
        v = x + u

        # run DnCNN denoiser 

        p = torch.from_numpy(v.transpose(2, 0, 1))
        p = p.type(torch.FloatTensor)
        p = p[None, ...].to(device)  # add batch dimension    
        v_tensor_denoised = model(p)        

        z = torch.squeeze(v_tensor_denoised).cpu().numpy().transpose(1, 2, 0)

        # u update
        u = u + x - z

    return x
