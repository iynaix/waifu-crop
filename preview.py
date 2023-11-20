import json
import cv2
from utils import (
    Box,
    WALLPAPER_DIR,
    detect,
    Cropper,
    draw,
    # FRAMEWORK_ASPECT_RATIO,
    VERTICAL_ASPECT_RATIO,
)


def preview_image(
    image,
    boxes: list[Box],
    idx: int,
    # (width, height)
    ratio: tuple[int, int] = (9, 16),
) -> int:
    rect, detection_boxes = Cropper(image, boxes, ratio).crop()
    boxes_to_draw = [rect, *detection_boxes]

    drawn_image = draw(
        image,
        boxes_to_draw,
        # BGR
        color=(0, 0, 255),
        thickness=3,
    )

    w, h = image.shape[:2][::-1]
    resized_image = cv2.resize(drawn_image, (1280, int(h / w * 1280)))
    cv2.imshow("Image", resized_image)

    key = cv2.waitKey(0) & 0xFF
    # esc
    if key == ord("q") or key == 27:
        # quit
        return 1000000
    # right arrow
    elif key == ord("n") or key == 39:
        return idx + 1
    # left arrow
    elif key == ord("p") or key == 37:
        return idx - 1

    return idx + 1


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

        # display the images
        idx = preview_image(
            image,
            boxes,
            idx,
            ratio=VERTICAL_ASPECT_RATIO,
            # ratio=FRAMEWORK_ASPECT_RATIO,
        )

    # write skipped images to file
    json.dump(sorted(images_to_skip), open("skip.json", "w"))
