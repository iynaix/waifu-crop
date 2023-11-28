import cv2
import shutil
import subprocess
from PIL import Image
from utils import (
    Cropper,
    WallpaperInfo,
    WALLPAPER_DIR,
    VERTICAL_ASPECT_RATIO,
    detect,
)
from pathlib import Path

INPUT_DIR = Path("in")

TARGET_WIDTH = 3440
TARGET_HEIGHT = 1504  # framework height


def crop_from_geometry(geometry: str, input: str, output: str) -> str:
    # split geometry into width, height, x, y
    w, h, x, y = [
        int(n) for n in geometry.replace("x", " ").replace("+", " ").split(" ")
    ]

    xmax = x + w
    ymax = y + h

    img = cv2.imread(input)
    cv2.imwrite(output, img[y:ymax, x:xmax])


if __name__ == "__main__":
    # create vertical wallpapers preview output directory
    PREVIEW_DIR = INPUT_DIR / "preview"
    IMAGE_DATA = WallpaperInfo()

    for p in sorted(d for d in INPUT_DIR.iterdir() if d.is_file()):
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

        # crop faces and write data
        faces = detect(str(out_path), face_score_threshold=0.5)
        image = cv2.imread(str(out_path))
        geometries = Cropper(image, faces).geometries()

        # output vertical image for preview
        if len(faces) > 1:
            PREVIEW_DIR.mkdir(exist_ok=True)

            vertical_str = f"{VERTICAL_ASPECT_RATIO[0]}x{VERTICAL_ASPECT_RATIO[1]}"
            crop_from_geometry(
                geometries[vertical_str],
                str(out_path),
                str(PREVIEW_DIR / p.name),
            )

        IMAGE_DATA[out_path.name] = geometries
        IMAGE_DATA.save()
