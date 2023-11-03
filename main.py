import json
import cv2
import anime_face_detector
from crop import (
    WALLPAPER_DIR,
    VERT_WALLPAPER_DIR,
    write_cropped_image,
    preview_image,
)


def detect(
    img,
    face_score_threshold: float,
    detector: anime_face_detector.LandmarkDetector,
):
    image = cv2.imread(img)
    preds = detector(image)

    boxes = []
    for pred in preds:
        box = pred["bbox"]
        score = box[4]
        if score < face_score_threshold:
            continue

        boxes.append(
            {
                "xmin": int(box[0]),
                "ymin": int(box[1]),
                "xmax": int(box[2]),
                "ymax": int(box[3]),
                "confidence": box[4],
            }
        )

    return boxes


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
        cv2.putText(
            image,
            label,
            (xmin, ymax - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            1,
        )
    return image


PREVIEW_IMAGES = False

if __name__ == "__main__":
    detector = anime_face_detector.create_detector(
        # faster-rcnn is also available
        face_detector_name="yolov3",
        # "cuda:0" is also available
        device="cpu",
    )

    try:
        images_to_skip = set(json.load(open("skip.json")))
    except FileNotFoundError:
        images_to_skip = set()

    # skip images if already cropped
    vertical_wallpapers = set(f.name for f in VERT_WALLPAPER_DIR.iterdir())
    image_paths = [
        img
        for img in WALLPAPER_DIR.iterdir()
        if img.name not in vertical_wallpapers and img.name not in images_to_skip
    ]

    # uncomment to test specific images
    # image_paths = (
    #     WALLPAPER_DIR / f
    #     for f in [
    #         "wallhaven-l8dj1p.jpg",
    #     ]
    # )

    if PREVIEW_IMAGES:
        print("Start inferencing. Press `q` to cancel. Press  `-` to go back.")

    image_paths = sorted(image_paths)

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
            detector=detector,
        )

        # skip if no boxes
        if not boxes:
            images_to_skip.add(path.name)
            idx += 1
            continue

        print(path, "x".join(image.shape[:2:-1]))

        # display the images
        if PREVIEW_IMAGES:
            idx = preview_image(image, boxes, idx)
        # write to file
        else:
            write_cropped_image(image, boxes, path.name)
            idx += 1

    # write skipped images to file
    json.dump(sorted(images_to_skip), open("skip.json", "w"))
