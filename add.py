import cv2
from utils import (
    WallpaperInfo,
    Cropper,
    WALLPAPER_DIR,
    box_to_geometry,
    SQUARE_ASPECT_RATIO,
)

# adds a new aspect ratio
if __name__ == "__main__":
    IMAGE_DATA = WallpaperInfo()
    for fname, info in sorted(IMAGE_DATA.data.items()):
        print(fname)

        cropper = Cropper(
            cv2.imread(str(WALLPAPER_DIR / fname)),
            [dict(zip(("xmin", "xmax", "ymin", "ymax"), f)) for f in info["faces"]],
            aspect_ratio=SQUARE_ASPECT_RATIO,
        )

        IMAGE_DATA[fname]["1x1"] = box_to_geometry(cropper.crop())
    IMAGE_DATA.save()
