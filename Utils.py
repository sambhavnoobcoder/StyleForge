from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm
from PIL import Image
import torch

class MingleModel:

    def __init__(self):
        # Set device
        self.torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load the autoencoder model which will be used to decode the latents into image space.
        use_auth_token = "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"
        self.vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae",
                                            use_auth_token=use_auth_token).to(self.torch_device)

        # Load the tokenizer and text encoder to tokenize and encode the text.
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", use_auth_token=use_auth_token)
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", use_auth_token=use_auth_token).to(self.torch_device)

        # # The UNet model for generating the latents.
        self.unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet",use_auth_token=use_auth_token).to(self.torch_device)

        # The noise scheduler
        self.scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                         num_train_timesteps=1000)

    def do_tokenizer(self, prompt):
        return self.tokenizer([prompt], padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True,
                  return_tensors="pt")

    def get_text_encoder(self, text_input):
        return self.text_encoder(text_input.input_ids.to(self.torch_device))[0]

    def latents_to_pil(self, latents):
        # bath of latents -> list of images
        latents = (1 / 0.18215) * latents
        with torch.no_grad():
            image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        return pil_images

    def generate_with_embs(self, text_embeddings, generator_int=32, num_inference_steps=30, guidance_scale=7.5):
        height = 512  # default height of Stable Diffusion
        width = 512  # default width of Stable Diffusion
        num_inference_steps = num_inference_steps  # Number of denoising steps
        guidance_scale = guidance_scale  # Scale for classifier-free guidance
        generator = torch.manual_seed(generator_int)  # Seed generator to create the inital latent noise
        batch_size = 1

        max_length = 77
        uncond_input = self.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.torch_device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # Prep Scheduler
        self.scheduler.set_timesteps(num_inference_steps)

        # Prep latents
        latents = torch.randn((batch_size, self.unet.in_channels, height // 8, width // 8), generator=generator)
        latents = latents.to(self.torch_device)
        latents = latents * self.scheduler.init_noise_sigma

        # Loop
        for i, t in tqdm(enumerate(self.scheduler.timesteps)):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            sigma = self.scheduler.sigmas[i]
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        return self.latents_to_pil(latents)[0]