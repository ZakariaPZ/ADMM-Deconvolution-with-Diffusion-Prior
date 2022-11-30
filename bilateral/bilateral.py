import numpy as np
from utils.fspecial import fspecial_gaussian_2d

def bilateral2d(img, sigma, sigmaIntensity):
    radius = int(sigma)
    pad = radius
    # Initialize filtered image to 0
    out = np.zeros_like(img)

    # Pad image to reduce boundary artifacts
    imgPad = np.pad(img, pad)

    # Smoothing kernel, gaussian with standard deviation sigma
    # and size (2*radius+1, 2*radius+1)
    filtSize = (2*radius + 1, 2*radius + 1)
    spatialKernel = fspecial_gaussian_2d(filtSize, sigma)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            centerVal = imgPad[y+pad, x+pad] # Careful of padding amount!

            window = imgPad[y:y+2*radius+1, x:x+2*radius+1] 
            # window = window*spatialKernel 
            intensityWindow = np.exp(-1*np.square(centerVal-window)/(2*((sigmaIntensity)**2)))

            IFiltered = np.sum(window*spatialKernel*intensityWindow)/np.sum(spatialKernel*intensityWindow)

            # Go over a window of size (2*radius + 1) around the current pixel,
            # compute weights, sum the weighted intensity.
            # Don't forget to normalize by the sum of the weights used.
            out[y, x] = IFiltered
    return out
