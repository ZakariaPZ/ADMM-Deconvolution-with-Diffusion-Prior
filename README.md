# Deconvolution using ADMM with Diffusion Denoising Prior

This project can be run with:

      python main.py --{method}  --sigma_noise {noise level} --img_name {file name}

Methods include: nlm, TV, bilateral, dncnn, diffusion

If no argument is provided, the default setting is to use all of them. 

### Example
To run the diffusion prior on a noise level of 0.1:
    
      python main.py --diffusion  --sigma_noise 0.1 --img_name face.png
