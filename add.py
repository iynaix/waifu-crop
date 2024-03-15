import cv2
from utils import (
    WallpaperInfo,
    Face,
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
            [
                Face(xmin=f["xmin"], xmax=f["xmax"], ymin=f["ymin"], ymax=f["ymax"])
                for f in info["faces"]
            ],
            aspect_ratio=SQUARE_ASPECT_RATIO,
        )

        IMAGE_DATA[fname]["r1x1"] = box_to_geometry(cropper.crop())
    IMAGE_DATA.save()
