import numpy as np
from numpy.fft import fft2, ifft2

import argparse
import skimage.io as io
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.transform import resize
import os
import datetime
import json

from pypher.pypher import psf2otf
import matplotlib.pyplot as plt

from utils.fspecial import *

from DnCNN.DnCNN import *
from TV.deconv_admm_tv import *
from NLM.deconv_admm_NLM import *
from DnCNN.deconv_admm_dncnn import *
from wiener.deconv_admm_wiener import *
from bilateral.deconv_admm_bilateral import *
from diffusion.deconv_admm_diffusion import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_name', type=str, default='face.png')
    parser.add_argument('-TV', '--TV', action= 'store_true')
    parser.add_argument('-wiener', '--wiener', action = 'store_true')
    parser.add_argument('-dncnn', '--dncnn', action = 'store_true')
    parser.add_argument('-bilateral', '--bilateral', action = 'store_true')
    parser.add_argument('-nlm', '--nlm', action = 'store_true')
    parser.add_argument('-diffusion', '--diffusion', action = 'store_true')
    parser.add_argument('--sigma_noise', type=float, default=0.1)
    parser.add_argument('--img_size', type=int, default=256)
 
    args = parser.parse_args()

    # create results folder and save config
    output_dir = f"./results/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/config.txt", 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # Select image
    img = io.imread(f'testimgs/{args.img_name}').astype(float)/255
    img = resize(img, (args.img_size, args.img_size))

    # blur kernel
    kernel_size = 30 if args.img_size==512 else 15
    kernel_sigma = 2.5 if args.img_size==512 else 1.25
    c = fspecial_gaussian_2d((kernel_size, kernel_size), kernel_sigma)

    # Blur kernel
    cFT = psf2otf(c, (img.shape[0], img.shape[1]))
    Afun = lambda x: np.real(ifft2(fft2(x) * cFT))

    # noise parameter - standard deviation
    sigma = args.sigma_noise

    print(sigma)
    # simulated measurements
    b = np.zeros(np.shape(img))
    for it in range(3):
        b[:, :, it] = Afun(img[:, :, it]) + sigma * np.random.randn(img.shape[0], img.shape[1])

    # save degraded image
    plt.imsave(f"{output_dir}/degraded.png", np.clip(b, 0, 1))

    all_condition = True
    for arg in vars(args):
        all_condition = all_condition and (not getattr(args, arg))
    

    if(args.TV or all_condition):
        # ADMM parameters for TV prior
        num_iters = 75
        rho = 5
        lam = 0.025

        # # # run ADMM+TV solver
        x_admm_tv = np.zeros(np.shape(b))
        for it in range(3):
            x_admm_tv[:, :, it] = deconv_admm_tv(b[:, :, it], c, lam, rho, num_iters)
        x_admm_tv = np.clip(x_admm_tv, 0.0, 1.0)
        PSNR_ADMM_TV = round(compute_psnr(img, x_admm_tv), 1)
        plt.imsave(f"{output_dir}/admm_tv_psnr{round(PSNR_ADMM_TV, 1)}.png", x_admm_tv)


    if(args.wiener or all_condition):
        # ADMM parameters for Wiener Prior
        num_iters = 75
        lam = 0.05
        rho = 1 * 0.5

        # # run ADMM+Wiener solver
        x_admm_wiener = np.zeros(np.shape(b))
        for it in range(3):
            x_admm_wiener[:, :, it] = deconv_admm_wiener(b[:, :, it], c, lam, rho, num_iters, sigma)
        x_admm_wiener = np.clip(x_admm_wiener, 0.0, 1.0)
        PSNR_ADMM_WIENER = round(compute_psnr(img, x_admm_wiener), 1)
        plt.imsave(f"{output_dir}/admm_wiener_psnr{round(PSNR_ADMM_WIENER, 1)}.png", x_admm_wiener)


    if(args.dncnn or all_condition):
        # Load DnCNN
        model = torch.load(f'models/DnCNN_{sigma}_{args.img_size}.pth')

        num_iters = 75
        lam = 0.05
        rho = 1 * 0.5

        # run DnCNN + ADMM
        x_admm_dncnn = deconv_admm_dncnn(b, c, lam, rho, num_iters,model)
        x_admm_dncnn = np.clip(x_admm_dncnn, 0, 1)
        b = np.clip(b, 0, 1)
        img = np.clip(img, 0, 1)
        PSNR_ADMM_DNCNN = round(compute_psnr(img, x_admm_dncnn), 1)
        plt.imsave(f"{output_dir}/admm_dncnn_psnr{round(PSNR_ADMM_DNCNN, 1)}.png", x_admm_dncnn)


    if(args.bilateral or all_condition):
        sigmaIntensity = 0.25

        # run ADMM+bilateral solver
        x_admm_bil = np.zeros(np.shape(b))
        for it in range(3):
            x_admm_bil[:, :, it] = deconv_admm_bilateral(b[:, :, it], c, lam, rho, num_iters, sigma, sigmaIntensity)
        x_admm_bil = np.clip(x_admm_bil , 0, 1)
        PSNR_ADMM_BIL = round(compute_psnr(img, x_admm_bil), 1)
        plt.imsave(f"{output_dir}/admm_bil_psnr{round(PSNR_ADMM_BIL, 1)}.png", x_admm_bil)



    if(args.nlm or all_condition):
        num_iters = 5
        lam = 0.01 * 0.5
        rho = 1 * 0.5
        nlmSigma = 0.1
        searchWindowRadius = 2

        # run ADMM+NLM solver
        x_admm_NLM = np.zeros(np.shape(b))
        for it in range(3):
            x_admm_NLM[:, :, it] = deconv_admm_NLM(b[:, :, it], c, lam, rho, num_iters, searchWindowRadius, sigma, nlmSigma)
        x_admm_NLM = np.clip(x_admm_NLM , 0, 1)
        PSNR_ADMM_NLM = round(compute_psnr(img, x_admm_NLM), 1)
        plt.imsave(f"{output_dir}/admm_nlm_psnr{round(PSNR_ADMM_NLM, 1)}.png", x_admm_NLM)


    if(args.diffusion or all_condition):
        num_iters = 10
        lam = 0.05
        rho = 1 * 0.5

        # run diffusion denoiser + ADMM
        x_admm_diffusion = deconv_admm_diffusion(b, c, lam, rho, num_iters, sigma)
        x_admm_diffusion = np.clip(x_admm_diffusion, 0, 1)
        img = np.clip(img, 0, 1)
        PSNR_ADMM_DIFFUSION = round(compute_psnr(img, x_admm_diffusion), 1)
        plt.imsave(f"{output_dir}/admm_diffusion_psnr{round(PSNR_ADMM_DIFFUSION, 1)}.png", x_admm_diffusion)