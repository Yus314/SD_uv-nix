from pathlib import Path
from diffusers import schedulers
import torch
from PIL import Image
from torch._C import _functionalization_reapply_views_tls
from util import gen_text_embeddings, load_image, model_load
from config import *
from sampler import perform_sampling
import numpy as np
import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

# モデルのロード
vae, tokenizer, text_encoder, unet, scheduler = model_load(device)
generator = torch.manual_seed(seed)


def process_image(image_path: Path, output_dir: Path, A, Ap, prompt):
    y = load_image(image_path, device)

    def get_mask_range(center, size):
        return center - size // 2, center + size // 2 - 1

    mask_stt_h, mask_end_h = get_mask_range(height // 2, mask_size)
    mask_stt_w, mask_end_w = get_mask_range(width // 2, mask_size)

    y[
        :,
        mask_stt_h:mask_end_h,
        mask_stt_w:mask_end_w,
    ] = 0

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
        A,
        Ap,
    )

    with torch.no_grad():
        image = vae.decode(latents)

    image = (image.sample / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    final_pil = np.array(Image.open(image_path).resize((height, width)), np.uint8)
    final_pil[
        mask_stt_h:mask_end_h,
        mask_stt_w:mask_end_w,
        :,
    ] = images[
        0,
        mask_stt_h:mask_end_h,
        mask_stt_w:mask_end_w,
        :,
    ]
    images = Image.fromarray(final_pil)

    # 画像を保存する
    output_path = output_dir / f"{image_path.stem}_result.png"
    images.save(output_path)


# def process_images_in_directory(input_dir: Path, output_dir: Path, A, Ap):
#    """ディレクトリ内のすべての画像を処理"""
#    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
#    output_dir.mkdir(parents=True, exist_ok=True)

#    for image_path in input_dir.iterdir():
#        if image_path.suffix.lower() in image_extensions and image_path.is_file():
#            process_image(image_path, output_dir, A, Ap)


def process_images_from_toml(toml_file: str):
    """TOMLファイルから画像を読み込み補完を行なう"""
    data = toml.load(toml_file)

    if not data:
        print(f"Warning: No data found in {toml_file}")
        return

    output_dir = Path(f"./Data/OUT/BSDS500/{mask_size}_lama/")
    output_dir.mkdir(parents=True, exist_ok=True)

    for info in tqdm.tqdm(data.values()):
        image_path = Path(info["imagepath"])
        prompt = info.get("description", "")

        if image_path.exists() and image_path.is_file():
            process_image(image_path, output_dir, A, Ap, prompt)
        else:
            print(f"Warning: {image_path} does not exist or is not a file.")

    print(f"Processing completed. Output saved in {output_dir}")
