import torch
import math
import warnings

from torch import Tensor
from diffusers import FluxPipeline


@torch.inference_mode()
def decode_imgs(latents, pipeline):
    imgs = (
        latents / pipeline.vae.config.scaling_factor
    ) + pipeline.vae.config.shift_factor
    imgs = pipeline.vae.decode(imgs)[0]
    imgs = pipeline.image_processor.postprocess(imgs, output_type="pil")
    return imgs


@torch.inference_mode()
def encode_imgs(imgs, pipeline, dtype):
    latents = pipeline.vae.encode(imgs).latent_dist.sample()
    latents = (
        latents - pipeline.vae.config.shift_factor
    ) * pipeline.vae.config.scaling_factor
    latents = latents.to(dtype=dtype)
    return latents


def _pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(
        batch_size, num_channels_latents, height // 2, 2, width // 2, 2
    )
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(
        batch_size, (height // 2) * (width // 2), num_channels_latents * 4
    )

    return latents


def _unpack_latents(latents, height, width, vae_scale_factor):
    batch_size, num_patches, channels = latents.shape

    # VAE applies 8x compression on images but we must also account for packing which requires
    # latent height and width to be divisible by 2.
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))

    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)

    latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

    return latents


def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = (
        latent_image_ids.shape
    )

    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )

    return latent_image_ids.to(device=device, dtype=dtype)


def get_timesteps_flux(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list:
    def get_lin_function(
        x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
    ):
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return lambda x: m * x + b

    def time_shift(mu: float, sigma: float, t: Tensor):
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1, dtype=torch.float32)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # estimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


class FluxWrapper:
    def __init__(
        self,
        pipeline: FluxPipeline,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        r"""Initialize the FluxWrapper.

        The constructor sets up the image dimensions, latent shapes, and initializes caches for prompt embeddings.

        Args:
            pipeline (`FluxPipeline`):
                The FluxPipeline object containing the diffusion model and associated components.
            height (`int`):
                The height of the images. Must be divisible by 16.
            width (`int`):
                The width of the images. Must be divisible by 16.
            dtype (`torch.dtype`):
                The data type for computations (e.g., `torch.float32`).
            device (`torch.device`):
                The device on which computations are performed (e.g., `torch.device('cuda')`).

        Raises:
            `Warning`:
                If `height` or `width` is not divisible by 16, they are adjusted to the nearest lower multiple of 16,
                and a warning is issued.
        """
        self.pipeline = pipeline

        if height % 16 != 0 or width % 16 != 0:
            height = 16 * (height // 16)
            width = 16 * (width // 16)
            warnings.warn(
                f"Height and width must be divisible by 16. Adjusted to {height}x{width}."
            )
        self.height, self.width = height, width
        self.vae_latent_shape = (16, height // 8, width // 8)  # C', H', W'
        self.dit_latent_shape = (
            height // 16 * width // 16,
            16 * 4,
        )  # T, C, used for pi_0 generation
        self.image_seq_len = (height // 16) * (width // 16)

        self.dtype = dtype
        self.device = device

        # Initialize cache for prompt embeddings
        self.cached_prompt = None
        self.prompt_embeds = None
        self.pooled_prompt_embeds = None
        self.text_ids = None

    def prepare_time_grid(
        self,
        num_steps: int,
        shift: bool = True,
    ):
        """Prepares the time grid for the diffusion process.

        Args:
            num_steps (`int`):
                The number of steps in the time grid.
            shift (`bool`, defaults to `True`):
                Whether to apply a time shift to the grid.
        """
        time_grid = get_timesteps_flux(
            num_steps=num_steps,
            image_seq_len=self.image_seq_len,
            shift=shift,
        )
        time_grid = [1.0 - t for t in time_grid]

        return time_grid

    def __call__(
        self,
        x_t: Tensor,
        t: torch.Tensor,
        prompt: str | None = None,
        guidance_scale: float = 3.5,
        prompt_embeds: torch.Tensor | None = None,
        pooled_prompt_embeds: torch.Tensor | None = None,
    ):
        """
        Compute the flux velocity for a given latent state and time.

        Args:
            x_t (`Tensor`):
                The packed latent variables at time `t`. Shape: `(batch_size, image_seq_len, latent_dim)`.
            t (`torch.Tensor`):
                The time in Rectified Flow ODE, which will be converted to Flux's time (`1 - t`).
                Shape: `(batch_size,)`.
            prompt (`str`, *optional*):
                The text prompt for conditional generation. If `prompt_embeds` and `pooled_prompt_embeds` are
                provided, this argument is ignored. Defaults to `None`.
            guidance_scale (`float`, *optional*, defaults to `3.5`):
                The guidance scale for Flux-dev model. Higher values encourage the model to better
                follow the prompt but may lead to quality degradation.
            prompt_embeds (`torch.Tensor`, *optional*):
                Precomputed prompt embeddings. If provided, `prompt` is ignored for embeddings. Defaults to `None`.
            pooled_prompt_embeds (`torch.Tensor`, *optional*):
                Precomputed pooled prompt embeddings. Must be provided if `prompt_embeds` is provided. Defaults to `None`.

        Returns:
            Tensor: The flux velocity corresponding to Rectified Flow ODE state `x_t` at time `t`.
        """
        if x_t.device != self.device:
            x_t = x_t.to(device=self.device)
            warnings.warn(
                f"x_t was moved to the device {self.device} of the FluxWrapper."
            )

        if x_t.dtype != self.dtype:
            x_t = x_t.to(dtype=self.dtype)
            warnings.warn(
                f"x_t was casted to the dtype {self.dtype} of the FluxWrapper."
            )

        # Convert ODE time t to Flux time 1 - t
        t_vec = 1.0 - t
        assert (
            isinstance(t_vec, torch.Tensor)
            and t_vec.ndim == 1
            and t.shape[0] == x_t.shape[0]
        ), "Time vector must be a 1D tensor with the same length as the batch size."

        # Prepare latent image ids
        latent_image_ids = _prepare_latent_image_ids(
            x_t.shape[0],
            self.vae_latent_shape[1] // 2,
            self.vae_latent_shape[2] // 2,
            self.device,
            self.dtype,
        )

        # Prepare guidance vector
        guidance_vec = torch.full(
            (x_t.shape[0],), guidance_scale, device=self.device, dtype=self.dtype
        )

        with torch.inference_mode():
            if prompt_embeds is not None and pooled_prompt_embeds is not None:
                self.prompt_embeds = prompt_embeds
                self.pooled_prompt_embeds = pooled_prompt_embeds
                _, _, self.text_ids = self.pipeline.encode_prompt(
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                )
            else:
                if prompt is None:
                    assert (
                        self.cached_prompt is not None
                    ), "Prompt must be provided if not cached."
                elif prompt != self.cached_prompt:  # Encode prompt if it has changed
                    self.cached_prompt = prompt
                    self.prompt_embeds, self.pooled_prompt_embeds, self.text_ids = (
                        self.pipeline.encode_prompt(
                            prompt=prompt,
                            prompt_2=prompt,
                        )
                    )
                    print(f'Prompt "{prompt}" encoded.')

        flux_velocity = self.pipeline.transformer(
            hidden_states=x_t,
            timestep=t_vec,
            guidance=guidance_vec,
            pooled_projections=self.pooled_prompt_embeds,
            encoder_hidden_states=self.prompt_embeds,
            txt_ids=self.text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=None,
            return_dict=self.pipeline,
        )[0]

        flux_velocity = -flux_velocity

        return flux_velocity

    @torch.inference_mode()
    def decode(
        self,
        dit_latents: Tensor,
    ):
        r"""Decode DiT latents into images.

        This method converts DiT latents (packed latents) back into images by first unpacking them into VAE latents
        and then decoding them using the VAE decoder.

        Args:
            dit_latents (`Tensor`):
                The DiT latents to decode. Shape: `(batch_size, image_seq_len, latent_dim)`.

        Returns:
            img_tensor (`Tensor`):
                The decoded images. Shape: `(batch_size, channels, height, width)`.
        """
        dit_latents = dit_latents.clone().to(device=self.device, dtype=self.dtype)
        assert (
            dit_latents.shape[1] == self.image_seq_len
        ), "Number of patches must match the image sequence length."
        assert (
            dit_latents.shape[2] == 16 * 4
        ), "Number of channels must match the VAE latent channels."
        latents = _unpack_latents(
            dit_latents, self.height, self.width, vae_scale_factor=8
        )
        images = decode_imgs(latents, self.pipeline)[0]
        return images

    @torch.inference_mode()
    def encode(
        self,
        images: Tensor,
    ):
        r"""Encode images into DiT latents.

        This method encodes images into VAE latents using the VAE encoder and then packs them into DiT latents
        suitable for processing by the Flux model.

        Args:
            images (`Tensor`):
                The images to encode. Shape: `(batch_size, 3, height, width)`, with pixel values in range `[0, 1]`.

        Returns:
            dit_latents (`Tensor`):
                The DiT latents (packed latents). Shape: `(batch_size, image_seq_len, latent_dim)`.
        """
        assert (
            images.shape[2] == self.height and images.shape[3] == self.width
        ), "Image dimensions must match the height and width of the FluxWrapper."
        vae_latents = encode_imgs(images, self.pipeline, self.dtype)
        dit_latents = _pack_latents(
            vae_latents,
            vae_latents.shape[0],
            16,
            vae_latents.shape[2],
            vae_latents.shape[3],
        )
        assert (
            dit_latents.shape[1] == self.image_seq_len
        ), "Number of patches must match the image sequence length."
        assert (
            dit_latents.shape[2] == 16 * 4
        ), "Number of channels must match the VAE latent channels."
        return dit_latents

    def dit_to_vae_latents(
        self,
        dit_latents: Tensor,
    ):
        r"""Convert DiT latents into VAE latents.

        This method unpacks DiT latents (packed latents) into VAE latents, which can then be decoded into images.

        Args:
            dit_latents (`Tensor`):
                The DiT latents to convert. Shape: `(batch_size, image_seq_len, latent_dim)`.

        Returns:
            vae_latents (`Tensor`):
                The VAE latents. Shape: `(batch_size, latent_channels, latent_height, latent_width)`.
        """
        return _unpack_latents(dit_latents, self.height, self.width, vae_scale_factor=8)

    def vae_to_dit_latents(
        self,
        vae_latents: Tensor,
    ):
        r"""Convert VAE latents into DiT latents.

        This method packs VAE latents into DiT latents suitable for processing by the Flux model.

        Args:
            vae_latents (`Tensor`):
                The VAE latents to convert. Shape: `(batch_size, latent_channels, latent_height, latent_width)`.

        Returns:
            dit_latents (`Tensor`):
                The DiT latents (packed latents). Shape: `(batch_size, image_seq_len, latent_dim)`.
        """
        return _pack_latents(
            vae_latents,
            vae_latents.shape[0],
            16,
            vae_latents.shape[2],
            vae_latents.shape[3],
        )


def get_packed_latent(
    height,
    width,
    batch_size: int,
    vae_latents: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
):
    # VAE applies 8x compression on images but we must also account for packing which requires
    # latent height and width to be divisible by 2.
    height = height // 8
    width = width // 8

    # VAE latent channels = 16, VAE latent resolution = resolution // 8
    # shape [num_samples, 16, resolution // 8, resolution // 8]
    shape = (batch_size, 16, height, width)
    # print("VAE compressed shape = ", shape)

    # num of tokens = (resolution // 8 * resolution // 8) / 4
    # shape [num_samples, (resolution // 8 * resolution // 8) / 4,  3]
    if vae_latents is not None:
        assert (
            vae_latents.shape[0] == batch_size
        ), "Batch size must match the number of samples."
        assert (
            vae_latents.shape[1] == shape[1]
        ), "Number of channels must match the VAE latent channels."
        assert (
            vae_latents.shape[2] == shape[2]
        ), "Height must match the VAE latent height."
        assert (
            vae_latents.shape[3] == shape[3]
        ), "Width must match the VAE latent width."
    else:
        vae_latents = torch.randn(
            shape, generator=generator, device=device, dtype=dtype
        )

    # After packing, shape [num_samples, (resolution // 16 * resolution // 16), 16 * 2 * 2]
    packed_latents = _pack_latents(vae_latents, batch_size, 16, height, width)
    # print("Packed latents shape = ", packed_latents.shape)

    latent_image_ids = _prepare_latent_image_ids(
        batch_size, height // 2, width // 2, device, dtype
    )
    # print("Latent image ids shape = ", latent_image_ids.shape)

    return packed_latents, latent_image_ids


def unpack_and_decode(
    pipeline,
    packed_latents: Tensor,
    height: int,
    width: int,
):
    latents = _unpack_latents(packed_latents, height, width, vae_scale_factor=8)
    imgs = decode_imgs(latents, pipeline)[0]
    return imgs
