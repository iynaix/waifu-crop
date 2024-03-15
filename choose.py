import cv2
import os
from pathlib import Path
from utils import (
    Cropper,
    VERTICAL_ASPECT_RATIO,
    WALLPAPER_DIR,
    WallpaperInfo,
    box_to_geometry,
    detect,
    iter_images,
)

INPUT_DIR = Path("in/preview")
BOX_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 255),
]

VALID_KEYS = "1234567890abcdefghijklmqrstuvwxyz"


def draw(image, faces, font_scale=3, thickness=1):
    """
    Draw boxes on the image. This function does not modify the image in-place.
    Args:
        image: A numpy BGR image.
        boxes: A list of dicts of {xmin, xmax, ymin, ymax}
        font_scale: Font size for the image number display.
        thickness: Thickness of the line.
    Returns:
        A drawn image.
    """
    image = image.copy()
    for idx, face in enumerate(faces):
        color = BOX_COLORS[idx % len(BOX_COLORS)]

        xmin, ymin, xmax, ymax = face["xmin"], face["ymin"], face["xmax"], face["ymax"]
        label = VALID_KEYS[idx].upper()
        ret, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 5)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)
        cv2.rectangle(
            image,
            (xmin, ymax - ret[1] - baseline),
            (xmin + ret[0], ymax),
            color,
            -1,
        )
        cv2.putText(
            image,
            label,
            (xmin, ymax - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            3,
        )
    return image


if __name__ == "__main__":
    # TODO: allow selecting for other aspect ratios?
    ratio = VERTICAL_ASPECT_RATIO

    # skip images if already cropped
    image_paths = sorted(iter_images(INPUT_DIR))
    IMAGE_DATA = WallpaperInfo()

    print("Start inferencing. Press `q` to cancel. Press  `-` to go back.")
    idx = 0
    while True:
        if idx >= len(image_paths) or idx < 0:
            break

        path = image_paths[idx]

        # use defaults
        fname = path.name
        wallpaper = str(WALLPAPER_DIR / fname)

        if not os.path.exists(wallpaper):
            wallpaper = wallpaper.replace(".jpg", ".png")

        image = cv2.imread(wallpaper)
        faces = detect(wallpaper)

        # skip if no faces
        if not faces:
            idx += 1
            continue

        # display the images
        cropper = Cropper(
            image,
            faces,
            # use aspect ratio from the input image?
            aspect_ratio=ratio,
        )
        rects = cropper.crop_candidates()
        drawn_image = draw(image, rects, thickness=3)

        w, h = image.shape[:2][::-1]
        resized_image = cv2.resize(drawn_image, (1280, int(h / w * 1280)))
        cv2.imshow("Image", resized_image)

        key = cv2.waitKey(0) & 0xFF
        # esc
        if key == ord("q") or key == 27:
            # quit
            idx = 1000000
        # right arrow
        elif key == ord("n") or key == 39:
            idx = idx + 1
        # left arrow
        elif key == ord("p") or key == 37:
            idx = idx - 1

        # crop the image on index selection
        elif key in [ord(c) for c in VALID_KEYS]:
            sel = VALID_KEYS.index(chr(key))
            rect = rects[sel]

            # update the data
            ratio_str = f"r{ratio[0]}x{ratio[1]}"
            IMAGE_DATA[fname][ratio_str] = box_to_geometry(rect)

            IMAGE_DATA.save()

            idx = idx + 1
        else:
            idx = idx + 1
