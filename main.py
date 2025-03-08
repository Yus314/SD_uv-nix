from config import input_dir, output_dir
from process import process_image


def main():
    if not input_dir.exists():
        print(f"Error: 入力ディレクトリ {input_dir} が存在しません")
        return

    process_image(input_dir, output_dir)


if __name__ == "__main__":
    main()
