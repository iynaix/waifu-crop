import json
import cv2
from crop import WALLPAPER_DIR, preview_image, detect


def draw(image, boxes, color=(0, 255, 0), font_scale=0.3, thickness=1):
    """
    Draw boxes on the image. This function does not modify the image in-place.
    Args:
        image: A numpy BGR image.
        boxes: A list of dicts of {xmin, xmax, ymin, ymax, confidence}
        colors: Color (BGR) used to draw.
        font_scale: Font size for the confidence level display.
        thickness: Thickness of the line.
    Returns:
        A drawn image.
    """
    image = image.copy()
    for box in boxes:
        xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
        confidence = box["confidence"]
        label = f"{confidence * 100:.2f}%"
        ret, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)
        cv2.rectangle(
            image,
            (xmin, ymax - ret[1] - baseline),
            (xmin + ret[0], ymax),
            color,
            -1,
        )
    return image


if __name__ == "__main__":
    try:
        images_to_skip = set(json.load(open("skip.json")))
    except FileNotFoundError:
        images_to_skip = set()

    # skip images if already cropped
    image_paths = [
        img for img in WALLPAPER_DIR.iterdir() if img.name not in images_to_skip
    ]

    # uncomment to test specific images
    # image_paths = sorted(Path("in").iterdir())

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
            images_to_skip.add(path.name)
            idx += 1
            continue

        print(path, "x".join(image.shape[:2:-1]))

        # display the images
        idx = preview_image(image, boxes, idx, ratio=(1440, 2560))
        # idx = preview_image(image, boxes, idx, ratio=(2256, 1506))

    # write skipped images to file
    json.dump(sorted(images_to_skip), open("skip.json", "w"))
