import torch
from inpainting import create_mask


def get_mask_range(center, size):
    return center - size // 2, center + size // 2 - 1


mask_size = 136
height = 512
width = 512
num_inference_steps = 30
guidance_scale = 7.5
seed = 18
batch_size = 1
eta = 0.85
data_set = "BSDS500"
prompt_gen = "noprompt"

mask_stt_h, mask_end_h = get_mask_range(height // 2, mask_size)
mask_stt_w, mask_end_w = get_mask_range(width // 2, mask_size)
if mask_size == 136:
    latent_mask_min = 23
    latent_mask_max = 41
elif mask_size == 168:
    latent_mask_min = 20
    latent_mask_max = 44
else:
    print("unexpected mask size")


# TOMLファイルから画像パスと説明文を読み込む
toml_file = f"./prompt_files/{data_set}_{mask_size}_{prompt_gen}.toml"

# デバイスの設定
device = "cuda" if torch.cuda.is_available() else "cpu"

# マスク
mask = create_mask(height, width, mask_size, latent_mask_min, latent_mask_max, device)


# AとApを関数として定義
def A(z):
    return z * mask


Ap = A
