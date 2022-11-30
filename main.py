import numpy as np
from numpy.fft import fft2, ifft2

import argparse
import skimage.io as io
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.filters import gaussian

from pypher.pypher import psf2otf
import matplotlib.pyplot as plt

from utils.fspecial import *

from DnCNN.DnCNN import *
from TV.deconv_admm_tv import *
from NLM.deconv_admm_NLM import *
from DnCNN.deconv_admm_dncnn import *
from wiener.deconv_admm_wiener import *
from bilateral.deconv_admm_bilateral import *

from dataset import BSDS300Dataset

# Select image
name = '3096'
img = io.imread(f'testimgs\{name}.jpg').astype(float)/255

# img = img.tran(img.shape[1], img.shape[2], 3)
# blur kernel
c = fspecial_gaussian_2d((30, 30), 2.5)

# Blur kernel
cFT = psf2otf(c, (img.shape[0], img.shape[1]))
Afun = lambda x: np.real(ifft2(fft2(x) * cFT))

# noise parameter - standard deviation
sigma = 0.1

# simulated measurements
b = np.zeros(np.shape(img))
for it in range(3):
    b[:, :, it] = Afun(img[:, :, it]) + sigma * np.random.randn(img.shape[0], img.shape[1])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-TV', '--TV', action= 'store_true')
    parser.add_argument('-wiener', '--wiener', action = 'store_true')
    parser.add_argument('-dncnn', '--dncnn', action = 'store_true')
    parser.add_argument('-bilateral', '--bilateral', action = 'store_true')
    parser.add_argument('-nlm', '--nlm', action = 'store_true')

    args = parser.parse_args()

    all_condition = True
    for arg in vars(args):
        all_condition = all_condition and (not getattr(args, arg))
    
    # # show images
    fig = plt.figure()

    if(args.TV or all_condition):
        # # # ADMM parameters for TV prior
        num_iters = 75
        rho = 5
        lam = 0.025

        # # # run ADMM+TV solver
        x_admm_tv = np.zeros(np.shape(b))
        for it in range(3):
            x_admm_tv[:, :, it] = deconv_admm_tv(b[:, :, it], c, lam, rho, num_iters)
        x_admm_tv = np.clip(x_admm_tv, 0.0, 1.0)
        PSNR_ADMM_TV = round(compute_psnr(img, x_admm_tv), 1)

        ax = fig.add_subplot(3, 2, 3)
        ax.imshow(x_admm_tv)
        ax.set_title("ADMM TV, PSNR: " + str(PSNR_ADMM_TV))
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    if(args.wiener or all_condition):
        # # ADMM parameters for DnCNN  & Wiener Prior
        num_iters = 75
        lam = 0.05
        rho = 1 * 0.5

        # # run ADMM+Wiener solver
        x_admm_wiener = np.zeros(np.shape(b))
        for it in range(3):
            x_admm_wiener[:, :, it] = deconv_admm_wiener(b[:, :, it], c, lam, rho, num_iters, sigma)
        x_admm_wiener = np.clip(x_admm_wiener, 0.0, 1.0)
        PSNR_ADMM_WIENER = round(compute_psnr(img, x_admm_wiener), 1)

        ax = fig.add_subplot(3, 2, 4)
        ax.imshow(x_admm_wiener)
        ax.set_title("ADMM Wiener, PSNR: " + str(PSNR_ADMM_WIENER))
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    if(args.dncnn or all_condition):
        # train DnCNN
        model = train(sigma=sigma, use_bias=True, hidden_channels=32)
        torch.save(model, 'DNCNN.pth')
        model = torch.load('DNCNN.pth')

        # run DnCNN + ADMM
        x_admm_dncnn = deconv_admm_dncnn(b, c, lam, rho, num_iters,model)
        x_admm_dncnn = np.clip(x_admm_dncnn, 0, 1)
        b = np.clip(b, 0, 1)
        img = np.clip(img, 0, 1)
        PSNR_ADMM_DNCNN = round(compute_psnr(img, x_admm_dncnn), 1)

        ax = fig.add_subplot(3, 2, 3)
        ax.imshow(x_admm_dncnn)
        ax.set_title("ADMM DnCNN, PSNR: " + str(PSNR_ADMM_DNCNN))
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    if(args.bilateral or all_condition):
        sigmaIntensity = 0.25
        # # run ADMM+bilateral solver
        x_admm_bil = np.zeros(np.shape(b))
        for it in range(3):
            x_admm_bil[:, :, it] = deconv_admm_bilateral(b[:, :, it], c, lam, rho, num_iters, sigma, sigmaIntensity)
        PSNR_ADMM_BIL = round(compute_psnr(img, x_admm_bil), 1)

        ax = fig.add_subplot(3, 2, 5)
        ax.imshow(x_admm_bil)
        ax.set_title("ADMM Bilateral, PSNR: " + str(PSNR_ADMM_BIL))
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    if(args.nlm or all_condition):
        # # Set up NLM

        num_iters = 5
        lam = 0.01 * 0.5
        rho = 1 * 0.5
        nlmSigma = 0.1  # Feel free to modify
        searchWindowRadius = 2

        # # run ADMM+NLM solver
        x_admm_NLM = np.zeros(np.shape(b))
        for it in range(3):
            x_admm_NLM[:, :, it] = deconv_admm_NLM(b[:, :, it], c, lam, rho, num_iters, searchWindowRadius, sigma, nlmSigma)
        PSNR_ADMM_NLM = round(compute_psnr(img, x_admm_NLM), 1)

        ax = fig.add_subplot(3, 2, 6)
        ax.imshow(x_admm_NLM)
        ax.set_title("ADMM NLM, PSNR: " + str(PSNR_ADMM_NLM))
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    ax = fig.add_subplot(3, 2, 1)
    ax.imshow(img)
    ax.set_title("Target Image")
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    ax = fig.add_subplot(3, 2, 2)
    ax.imshow(b)
    ax.set_title("Blurry and Noisy Image")
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    plt.show()