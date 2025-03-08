import torch
from torch import autocast
from sch_plus import randn_tensor


def perform_sampling(
    latents,
    scheduler,
    unet,
    text_embeddings,
    mask,
    latents_y,
    guidance_scale,
    eta,
    generator,
    device,
    num_inference_steps,
):
    with autocast("cuda"):
        for i, t in enumerate(scheduler.timesteps):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            with torch.no_grad():
                noise_pred = unet(
                    latent_model_input, t, encoder_hidden_states=text_embeddings
                )["sample"]

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            prev_timestep = (
                t
                - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
            )
            scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)
            alpha_prod_t = scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = (
                scheduler.alphas_cumprod[prev_timestep]
                if prev_timestep >= 0
                else scheduler.final_alpha_cumprod
            )
            beta_prod_t = 1 - alpha_prod_t

            pred_original_sample = (
                latents - beta_prod_t ** (0.5) * noise_pred
            ) / alpha_prod_t ** (0.5)
            pred_epsilon = noise_pred

            variance = scheduler._get_variance(t, prev_timestep)
            std_dev_t = eta * variance ** (0.5)

            pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (
                0.5
            ) * pred_epsilon

            prev_origiral_sample = mask * latents_y + (1 - mask) * pred_original_sample
            prev_sample = (
                alpha_prod_t_prev ** (0.5) * pred_original_sample
                + pred_sample_direction
            )

            variance_noise = randn_tensor(
                noise_pred.shape,
                generator=generator,
                device=noise_pred.device,
                dtype=noise_pred.dtype,
            )
            variance = std_dev_t * variance_noise
            prev_sample = prev_sample + variance

            latents = prev_sample

    latents = latents.float()
    latents = 1 / 0.18215 * latents
    return latents
