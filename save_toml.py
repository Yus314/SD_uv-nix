import toml
from pathlib import Path


def save_to_toml(data, filename):
    """画像の説明とパスをTOMLファイルに保存"""
    filepath = Path(filename)

    if filepath.exists():
        with open(filename, "r") as f:
            exsinting_data = toml.load(f)
    else:
        exsinting_data = {}

    exsinting_data.update(data)

    with open(filename, "w") as f:
        toml.dump(exsinting_data, f)
