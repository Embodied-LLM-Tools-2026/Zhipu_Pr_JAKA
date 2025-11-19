# resize_image.py
from PIL import Image, ImageOps
from pathlib import Path
import argparse

def resize_to_size(
    image_path: str,
    out_w: int,
    out_h: int,
    suffix: str = "_resized",
) -> str:
    """
    读取图片(保持原图EXIF朝向)，按指定(W,H)缩放(不保留比例)，
    保存到原文件夹，文件名加后缀，返回输出路径。
    """
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"No such file: {image_path}")

    # 读原图并根据EXIF旋转到正确朝向
    img = Image.open(p)
    img = ImageOps.exif_transpose(img)

    # 使用高质量重采样
    try:
        RESAMPLE = Image.Resampling.LANCZOS
    except AttributeError:
        RESAMPLE = getattr(Image, "LANCZOS", getattr(Image, "ANTIALIAS", Image.BICUBIC))

    img_resized = img.resize((int(out_w), int(out_h)), RESAMPLE)

    out_path = p.with_name(f"{p.stem}{suffix}{p.suffix}")
    img_resized.save(out_path)
    return str(out_path)

def main():
    ap = argparse.ArgumentParser(description="Resize image to specific size and save with suffix in the same folder.")
    ap.add_argument("--image", "-i", required=True, help="输入图片路径")
    ap.add_argument("--width", "-W", type=int, required=True, help="目标宽度（像素）")
    ap.add_argument("--height", "-H", type=int, required=True, help="目标高度（像素）")
    ap.add_argument("--suffix", "-s", default="_resized", help="输出文件名后缀，默认 _resized")
    args = ap.parse_args()

# python3 ./examples/resize.py --image /home/sht/DIJA/Photo/1.jpg --width 1000 --height 750 --suffix _1000x750

    out_path = resize_to_size(args.image, args.width, args.height, args.suffix)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
