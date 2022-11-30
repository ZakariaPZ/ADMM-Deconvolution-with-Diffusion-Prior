import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import skimage.io
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset import BSDS300Dataset

# set random seeds
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DnCNN(nn.Module):
    """
    Network architecture from this reference. Note that we omit batch norm
    since we are using a shallow network to speed up training times.

    @article{zhang2017beyond,
      title={Beyond a {Gaussian} denoiser: Residual learning of deep {CNN} for image denoising},
      author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
      journal={IEEE Transactions on Image Processing},
      year={2017},
      volume={26},
      number={7},
      pages={3142-3155},
    }
    """

    def __init__(self, in_channels=3, out_channels=3, hidden_channels=32, kernel_size=3,
                 hidden_layers=3, use_bias=True):
        super(DnCNN, self).__init__()

        self.use_bias = use_bias

        layers = []
        layers.append(torch.nn.Conv2d(in_channels, hidden_channels, kernel_size, padding='same', bias=use_bias))
        layers.append(torch.nn.ReLU(inplace=True))
        for i in range(hidden_layers):
            layers.append(torch.nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding='same', bias=use_bias))
            layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Conv2d(hidden_channels, out_channels, kernel_size, padding='same', bias=use_bias))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return x - self.net(x)


def add_noise(x, sigma=0.1):
    return x + torch.randn_like(x) * sigma


def img_to_numpy(x):
    return np.clip(x.detach().cpu().numpy().squeeze().transpose(1, 2, 0), 0, 1)


def calc_psnr(x, gt):
    out = 10 * np.log10(1 / ((x - gt)**2).mean().item())
    return out


def plot_summary(idx, model, sigma, losses, psnrs, baseline_psnrs,
                 val_losses, val_psnrs, val_iters, train_dataset,
                 val_dataset, val_dataloader):

    with torch.no_grad():
        model.eval()

        # evaluate on training dataset sample
        train_dataset.use_patches = False
        train_image = train_dataset[0][None, ...].to(device)
        train_dataset.use_patches = True

        noisy_train_image = add_noise(train_image, sigma=sigma)
        denoised_train_image = model(noisy_train_image)

        # evaluate on validation dataset sample
        val_dataset.use_patches = False
        val_image = val_dataset[6][None, ...].to(device)
        val_dataset.use_patches = True
        val_patch_samples = next(iter(val_dataloader)).to(device)

        # calculate validation metrics
        noisy_val_patch_samples = add_noise(val_patch_samples, sigma=sigma)
        denoised_val_patch_samples = model(noisy_val_patch_samples)
        val_loss = torch.mean((val_patch_samples - denoised_val_patch_samples)**2)
        val_psnr = calc_psnr(denoised_val_patch_samples, val_patch_samples)

        val_losses.append(val_loss.item())
        val_psnrs.append(val_psnr)
        val_iters.append(idx)

        noisy_val_image = add_noise(val_image, sigma=sigma)
        denoised_val_image = model(noisy_val_image)

    plt.clf()
    plt.subplot(241)
    plt.plot(losses, label='Train loss')
    plt.plot(val_iters, val_losses, '.', label='Val. loss')
    plt.yscale('log')
    plt.legend()
    plt.title('loss')

    plt.subplot(245)
    plt.plot(psnrs, label='Train PSNR')
    plt.plot(val_iters, val_psnrs, '.', label='Val. PSNR')
    plt.plot(baseline_psnrs, label='Baseline PSNR')
    plt.ylim((0, 32))
    plt.legend()
    plt.title('psnr')

    plt.subplot(242)
    plt.imshow(img_to_numpy(train_image))
    plt.ylabel('Training Set')
    plt.title('GT')

    plt.subplot(243)
    plt.imshow(img_to_numpy(noisy_train_image))
    plt.title('Noisy Image')

    plt.subplot(244)
    plt.imshow(img_to_numpy(denoised_train_image))
    plt.title('Denoised Image')

    plt.subplot(246)
    plt.imshow(img_to_numpy(val_image))
    plt.ylabel('Validation Set')
    plt.title('GT')

    plt.subplot(247)
    plt.imshow(img_to_numpy(noisy_val_image))
    plt.title('Noisy Image')

    plt.subplot(248)
    plt.imshow(img_to_numpy(denoised_val_image))
    plt.title('Denoised Image')

    plt.tight_layout()
    plt.pause(0.1)


def train(sigma=0.1, use_bias=True, hidden_channels=32, epochs=2, batch_size=32, plot_every=200):

    print(f'==> Training on noise level {sigma:.02f} | use_bias: {use_bias} | hidden_channels: {hidden_channels}')

    # create datasets
    train_dataset = BSDS300Dataset(patch_size=32, split='train', use_patches=True)
    val_dataset = BSDS300Dataset(patch_size=32, split='test', use_patches=True)

    # create dataloaders & seed for reproducibility
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = DnCNN(use_bias=use_bias, hidden_channels=hidden_channels).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    psnrs = []
    baseline_psnrs = []
    val_losses = []
    val_psnrs = []
    val_iters = []
    idx = 0

    pbar = tqdm(total=len(train_dataset) * epochs // batch_size)
    for epoch in range(epochs):
        for sample in train_dataloader:

            model.train()
            sample = sample.to(device)

            # add noise
            noisy_sample = add_noise(sample, sigma=sigma)

            # denoise
            denoised_sample = model(noisy_sample)

            # loss function
            loss = torch.mean((denoised_sample - sample)**2)
            psnr = calc_psnr(denoised_sample, sample)
            baseline_psnr = calc_psnr(noisy_sample, sample)

            losses.append(loss.item())
            psnrs.append(psnr)
            baseline_psnrs.append(baseline_psnr)

            # update model
            loss.backward()
            optim.step()
            optim.zero_grad()

            # plot results
            if not idx % plot_every:
                plot_summary(idx, model, sigma, losses, psnrs, baseline_psnrs,
                             val_losses, val_psnrs, val_iters, train_dataset,
                             val_dataset, val_dataloader)

            idx += 1
            pbar.update(1)

    pbar.close()
    return model


def evaluate_model(model, sigma=0.1, output_filename='out.png'):
    dataset = BSDS300Dataset(patch_size=32, split='test', use_patches=False)
    model.eval()

    psnrs = []
    for idx, image in enumerate(dataset):
        image = image[None, ...].to(device)  # add batch dimension
        noisy_image = add_noise(image, sigma)
        denoised_image = model(noisy_image)
        psnr = calc_psnr(denoised_image, image)
        psnrs.append(psnr)

        # include the tiger image in your homework writeup
        if idx == 6:
            skimage.io.imsave(output_filename, (img_to_numpy(denoised_image)*255).astype(np.uint8))

    return np.mean(psnrs)
