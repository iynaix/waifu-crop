import cv2
import shutil
import subprocess
from PIL import Image
from utils import (
    Cropper,
    FRAMEWORK_WALLPAPER_DIR,
    VERT_WALLPAPER_DIR,
    WALLPAPER_DIR,
    detect,
)
from pathlib import Path

INPUT_DIR = Path("in")

TARGET_WIDTH = 3440
TARGET_HEIGHT = 1504  # framework height


if __name__ == "__main__":
    for p in sorted(INPUT_DIR.iterdir()):
        img = Image.open(p)
        width, height = img.size

        needs_upscale = not (width >= TARGET_WIDTH and height >= TARGET_HEIGHT)

        if needs_upscale:
            out_path = WALLPAPER_DIR / (
                p.name.replace(".jpg", ".png").replace(".jpeg", ".png")
            )
            width2 = width * 2
            height2 = height * 2

            subprocess.run(
                [
                    "realesrgan-ncnn-vulkan",
                    "-i",
                    p,
                    "-n",
                    "realesrgan-x4plus-anime",
                    "-o",
                    out_path,
                ]
            )

            needs_resize = width2 >= TARGET_WIDTH and height2 >= TARGET_HEIGHT
            if needs_resize:
                img = Image.open(out_path)
                img = img.resize((width2, height2), Image.LANCZOS)
                img.save(out_path)
        else:
            # copy to output dir
            out_path = WALLPAPER_DIR / p.name
            shutil.copy(p, WALLPAPER_DIR / p.name)

        # optimize png
        if needs_upscale or p.suffix == ".png":
            subprocess.run(["oxipng", "--opt", "max", out_path])

        # crop faces
        if boxes := detect(
            str(out_path),
            face_score_threshold=0.5,
        ):
            image = cv2.imread(str(out_path))

            # write for vertical monitor
            Cropper(image, boxes, aspect_ratio=(1440, 2560)).write_cropped_image(
                str(VERT_WALLPAPER_DIR / out_path.name)
            )

            # write for framework
            Cropper(image, boxes, aspect_ratio=(2256, 1504)).write_cropped_image(
                str(FRAMEWORK_WALLPAPER_DIR / out_path.name)
            )
        else:
            # copy for framework
            shutil.copy(p, FRAMEWORK_WALLPAPER_DIR / p.name)
