import cv2
from pathlib import Path
from utils import (
    Cropper,
    FRAMEWORK_WALLPAPER_DIR,
    VERT_WALLPAPER_DIR,
    detect,
    VERTICAL_ASPECT_RATIO,
    FRAMEWORK_ASPECT_RATIO,
)

INPUT_DIR = Path("in")
BOX_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 255),
]
WRITE_VERTICAL = False


def draw(image, boxes, font_scale=3, thickness=1):
    """
    Draw boxes on the image. This function does not modify the image in-place.
    Args:
        image: A numpy BGR image.
        boxes: A list of dicts of {xmin, xmax, ymin, ymax, confidence}
        font_scale: Font size for the confidence level display.
        thickness: Thickness of the line.
    Returns:
        A drawn image.
    """
    image = image.copy()
    for idx, box in enumerate(boxes):
        color = BOX_COLORS[idx % len(BOX_COLORS)]

        xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
        label = str(idx + 1)
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
    # skip images if already cropped
    image_paths = sorted(INPUT_DIR.iterdir())

    print("Start inferencing. Press `q` to cancel. Press  `-` to go back.")
    idx = 0
    while True:
        if idx >= len(image_paths) or idx < 0:
            break

        path = image_paths[idx]

        # use defaults
        image = cv2.imread(str(path))
        boxes = detect(
            str(path),
            face_score_threshold=0.5,
        )

        # skip if no boxes
        if not boxes:
            idx += 1
            continue

        print(path, "x".join(image.shape[:2:-1]))

        # display the images
        aspect_ratio = (
            VERTICAL_ASPECT_RATIO if WRITE_VERTICAL else FRAMEWORK_ASPECT_RATIO
        )
        OUTPUT_DIR = VERT_WALLPAPER_DIR if WRITE_VERTICAL else FRAMEWORK_WALLPAPER_DIR

        cropper = Cropper(image, boxes, aspect_ratio=aspect_ratio)
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
        elif key in [ord(str(n)) for n in range(1, len(rects) + 1)]:
            cropper.write_cropped_image(
                str(OUTPUT_DIR / path.name), rect=rects[int(chr(key)) - 1]
            )
            idx = idx + 1
        else:
            idx = idx + 1
