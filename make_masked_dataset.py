from PIL import Image
from config import *
import numpy as np
import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"


def apply_mask(image_path):
    y = Image.open(image_path).convert("RGB").resize((512, 512))
    y = np.array(y)

    def get_mask_range(center, size):
        return center - size // 2, center + size // 2 - 1

    mask_stt_h, mask_end_h = get_mask_range(height // 2, mask_size)
    mask_stt_w, mask_end_w = get_mask_range(width // 2, mask_size)
    y[mask_stt_h:mask_end_h, mask_stt_w:mask_end_w, :] = 0

    images = Image.fromarray(y)

    # 画像を保存する
    output_path = output_dir / f"{image_path.stem}.png"
    images.save(output_path)


for image_path in tqdm.tqdm(input_dir.iterdir()):
    apply_mask(image_path)
