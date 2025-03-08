from pathlib import Path
from diffusers import schedulers
import torch
from PIL import Image
from util import gen_text_embeddings, load_image, model_load
from config import *
from inpainting import create_mask
from sampler import perform_sampling

device = "cuda" if torch.cuda.is_available() else "cpu"

# モデルのロード
vae, tokenizer, text_encoder, unet, scheduler = model_load(device)
generator = torch.manual_seed(seed)

# マスク作成
mask = create_mask(height, width, mask_size, latent_mask_min, latent_mask_max, device)


def A(z):
    return z * mask


Ap = A


def process_image(image_path: Path, output_dir: Path):
    y = load_image(image_path, device)
    y = torch.unsqueeze(y, dim=0)
    latents_y = 0.1825 * vae.encode(2 * y - 1).latent_dist.mean

    text_embeddings = gen_text_embeddings(
        tokenizer, text_encoder, device, batch_size, prompt
    )
    scheduler.set_timesteps(num_inference_steps)

    latents = (
        torch.randn(
            (batch_size, unet.config.in_channels, height // 8, width // 8),
            generator=generator,
        ).to(device)
        * scheduler.init_noise_sigma
    )

    latents = perform_sampling(
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
    )

    with torch.no_grad():
        image = vae.decode(latents)

    image = (image.sample / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    images = Image.fromarray(images.squeeze())

    # 画像を保存する
    output_path = output_dir / f"{image_path.stem}_result.png"
    images.save(output_path)
