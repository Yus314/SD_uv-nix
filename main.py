from config import toml_file
from process import process_images_from_toml


def main():
    process_images_from_toml(toml_file)


if __name__ == "__main__":
    main()
