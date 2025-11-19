from PIL import Image, ImageDraw
import argparse, os

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def compute_box(mode, nums, W, H):
    if mode == "xywh":
        if len(nums) != 4:
            raise ValueError("xywh requires 4 numbers: x y w h")
        x, y, w, h = nums
        x2, y2 = x + w, y + h
        x1, y1 = x, y
    elif mode == "xyxy":
        if len(nums) != 4:
            raise ValueError("xyxy requires 4 numbers: x1 y1 x2 y2")
        x1, y1, x2, y2 = nums
    else:
        raise ValueError("Unknown mode: %s" % mode)

    # Normalize so x1<=x2, y1<=y2
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1

    # Clamp to image bounds
    x1 = clamp(int(round(x1)), 0, W - 1)
    y1 = clamp(int(round(y1)), 0, H - 1)
    x2 = clamp(int(round(x2)), 0, W - 1)
    y2 = clamp(int(round(y2)), 0, H - 1)
    return x1, H-y1, x2, H-y2

def draw_rectangle(img_path, numbers, mode="xywh", thickness=3, color=(255,0,0), out_path=None):
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    x1, y1, x2, y2 = compute_box(mode, numbers, W, H)

    draw = ImageDraw.Draw(img)
    # Simulate thickness by drawing multiple rectangles
    for t in range(thickness):
        draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=color)

    if out_path is None:
        root, ext = os.path.splitext(img_path)
        out_path = f"{root}_rect{ext or '.png'}"
    img.save(out_path)
    return out_path, (x1, y1, x2, y2), (W, H)

def main():
    ap = argparse.ArgumentParser(description="Draw a red rectangle on an image given 4 numbers.")
    ap.add_argument("--image", "-i", required=True, help="Path to input image")
    ap.add_argument("--numbers", "-n", nargs=4, type=float, required=True,
                    help="Four numbers; use --mode to interpret them")
    ap.add_argument("--mode", choices=["xywh","xyxy"], default="xywh",
                    help="xywh = x,y,width,height; xyxy = x1,y1,x2,y2")
    ap.add_argument("--thickness", type=int, default=3, help="Rectangle border thickness (px)")
    ap.add_argument("--output", "-o", default=None, help="Path to save the result")
    args = ap.parse_args()

    out_path, box, size = draw_rectangle(args.image, args.numbers, args.mode, args.thickness, (255,0,0), args.output)
    print(f"Saved: {out_path}")
    print(f"Image size: {size[0]}x{size[1]}  Box ({args.mode} normalized to xyxy): {box}")

if __name__ == "__main__":
    main()