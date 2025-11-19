# Create a reusable demo script and run a quick example.
# The script supports two rectangle formats:
#   --mode xywh : x, y, width, height
#   --mode xyxy : x1, y1, x2, y2
# It draws a red rectangle and saves the result.

from PIL import Image, ImageDraw, ImageFont
import argparse, os, sys, math
import grounding as mod  

sample_name = "/home/sht/DIJA/Pr/UI/static/captures/front_1761052858639"
sample_path = sample_name + ".jpg"

out_path2 = sample_name + "_rect_xyxy.jpg"

out_path2, box2, size2 = mod.draw_rectangle(sample_path, [439, 605, 472, 721], mode="xyxy", thickness=5, color=(255,0,0), out_path=out_path2)
print(f"Saved: {out_path2} with box {box2} in image size {size2}")