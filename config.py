from pathlib import Path
import torch
from inpainting import create_mask

mask_size = 136
height = 512
width = 512
num_inference_steps = 30
guidance_scale = 7.5
seed = 18
batch_size = 1
eta = 0.85
latent_mask_min = 23
latent_mask_max = 41
prompt = [""]

# ディレクリ設定
input_dir = Path("./BSDS500/")
output_dir = Path("./output_images/")

# デバイスの設定
device = "cuda" if torch.cuda.is_available() else "cpu"

# マスク
mask = create_mask(height, width, mask_size, latent_mask_min, latent_mask_max, device)


# AとApを関数として定義
def A(z):
    return z * mask


Ap = A
