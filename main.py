from config import input_dir, output_dir, A, Ap
from process import process_images_in_directory


def main():
    if not input_dir.exists():
        print(f"Error: 入力ディレクトリ {input_dir} が存在しません")
        return

    process_images_in_directory(input_dir, output_dir, A, Ap)


if __name__ == "__main__":
    main()
