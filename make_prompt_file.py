import tqdm
from generate_description import generate_description
from pathlib import Path
from save_toml import save_to_toml

input_dirs = [
    "./Data/IN/Masked/BSDS500_168/",
    "./Data/IN/Masked/imagenet_1k_135/",
    "./Data/IN/Masked/imagenet_1k_168/",
    "./Data/IN/Masked/imagenet_o_135/",
    "./Data/IN/Masked/imagenet_o_168/",
]
output_files = [
    "./BSDS500_168.toml",
    "./imagenet_1k_135.toml",
    "./imagenet_1k_168.toml",
    "./imagenet_o_135.toml",
    "./imagenet_o_168.toml",
]

for input_dir, output_file in zip(input_dirs, output_files):
    data = {}
    input_path = Path(input_dir)

    if not input_path.exists():
        print(f"Warning: {input_dir} does not exist; Skipping...")
        continue
    for image_path in tqdm.tqdm(input_path.iterdir()):
        description = generate_description(image_path)
        data[str(image_path)] = {
            "imagepath": str(image_path),
            "description": description,
        }
        print(f"Processed: {image_path}")
        save_to_toml(data, output_file)
    print(f"Saved: to {output_file}")
