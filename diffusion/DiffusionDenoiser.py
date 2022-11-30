from diffusers import DDPMPipeline
import torch


class DDPMDenoiser(DDPMPipeline):
    @torch.no_grad()
    def __call__(
        self,
        init_image,
        init_t: int = 0,
        batch_size: int = 1,
        num_inference_steps: int = 1000,
    ):

        # Sample gaussian noise to begin loop
        if isinstance(self.unet.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.in_channels,
                self.unet.sample_size,
                self.unet.sample_size,
            )
        else:
            image_shape = (batch_size, self.unet.in_channels, *self.unet.sample_size)

        assert init_image.shape == image_shape
        image = init_image.clone().detach().to(self.device)
        image = (image - 0.5) * 2

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps[-init_t:]):
            # 1. predict noise model_output
            model_output = self.unet(image, t).sample

            # 2. compute previous image: x_t -> x_t-1
            image = self.scheduler.step(
                model_output, t, image, generator=None
            ).prev_sample

        image = (image / 2 + 0.5).clamp(0, 1)

        return image

    def get_init_timestep_from_variance(self, var):
        om_alphab = 1 - self.scheduler.alphas_cumprod
        idx = (torch.abs(om_alphab - var)).argmin()
        return idx
