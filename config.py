from pathlib import Path

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
prompt = ["a photo of kangaroo"]

# ディレクリ設定
input_dir = Path("./69020.jpg")
output_dir = Path("./output_images/")
