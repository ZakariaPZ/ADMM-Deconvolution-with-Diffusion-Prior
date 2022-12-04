from .DiffusionDenoiser import DDPMDenoiser
import numpy as np
from tqdm import tqdm
from numpy.fft import fft2, ifft2
from pypher.pypher import psf2otf
import torch
from skimage.transform import resize

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def deconv_admm_diffusion(b, c, lam, rho, num_iters, noise_variance):

    # initialize x,z,u with all zeros
    x = np.zeros_like(b)
    z = np.zeros_like(b)
    u = np.zeros_like(b)
    v = np.zeros_like(b)

    # set up DDPM
    model = DDPMDenoiser.from_pretrained("google/ddpm-celebahq-256")
    model = model.to(device)

    num_channels = b.shape[2]
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
            x[:, :, channel] = np.real(ifft2((cTFT * bFT + rho * vFT) / denom))

        # z update
        v = x + u

        # run diffusion denoiser
        v_resized = resize(v, (256, 256))
        v_tensor = torch.from_numpy(v_resized[None, :]).permute(0, 3, 1, 2).float().to(device)

        init_t = model.get_init_timestep_from_variance(var=noise_variance)
        v_tensor_denoised = model(init_image=v_tensor.clamp(0, 1), init_t=init_t)

        z = torch.squeeze(v_tensor_denoised).permute(1, 2, 0).cpu().numpy()
        z = resize(z, (v.shape[0], v.shape[0]))

        # u update
        u = u + x - z

    return x
