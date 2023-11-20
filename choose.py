import cv2
from pathlib import Path
from collections import defaultdict
from crop import (
    Box,
    BoxIntersections,
    AspectRatio,
    VERT_WALLPAPER_DIR,
    detect,
    get_largest_crop,
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


def calculate_crops(
    image,
    boxes: list[Box],
    # (width, height)
    ratio: AspectRatio = (9, 16),
) -> tuple[Box, list[Box]]:
    height, width = image.shape[:2]
    (target_width, target_height), direction = get_largest_crop((width, height), ratio)

    def clamp(val, direction):
        min_ = int(val)
        empty = {
            "xmin": 0,
            "xmax": 0,
            "ymin": 0,
            "ymax": 0,
            "confidence": 1,
        }

        # check if out of bounds and constrain it
        if direction == "x":
            max_ = min_ + target_width
            if min_ < 0:
                return {**empty, "xmax": target_width, "ymax": height}
            elif max_ > width:
                return {
                    **empty,
                    "xmin": width - target_width,
                    "xmax": width,
                    "ymax": height,
                }
            else:
                return {**empty, "xmin": min_, "xmax": max_, "ymax": height}
        else:
            max_ = min_ + target_height
            if min_ < 0:
                return {**empty, "ymax": target_height, "xmax": width}
            elif max_ > width:
                return {
                    **empty,
                    "ymin": height - target_height,
                    "ymax": width,
                    "xmax": width,
                }
            else:
                return {**empty, "ymin": min_, "ymax": max_, "xmax": width}

    if len(boxes) == 1:
        box = boxes[0]
        box_mid = (box[f"{direction}min"] + box[f"{direction}max"]) / 2
        target = (
            box_mid - target_width / 2
            if direction == "x"
            else box_mid - target_height / 2
        )

        box_new = {
            "xmin": 0,
            "xmax": 0,
            "ymin": 0,
            "ymax": 0,
            "confidence": box["confidence"],
            **clamp(target, direction),
        }

        return (
            box_new,
            boxes,
        )

    else:
        min_ = "xmin" if direction == "x" else "ymin"
        max_ = "xmax" if direction == "x" else "ymax"

        # sort boxes by min_
        boxes.sort(key=lambda box: box[min_])

        # (area, xmin of box)
        boxes_info: list[BoxIntersections] = []

        for rect_start in range(
            width - target_width if direction == "x" else height - target_height
        ):
            rect_end = rect_start + (
                target_width if direction == "x" else target_height
            )

            # check number of boxes in decimal within enclosed within larger rectangle
            boxes_area = 0
            for box in boxes:
                # no intersection, we overshot the final box
                if box[min_] > rect_end:
                    break

                # no intersection
                elif box[max_] < rect_start:
                    continue

                # full intersection
                elif box[min_] >= rect_start and box[max_] <= rect_end:
                    boxes_area += (box["xmax"] - box["xmin"]) * (
                        box["ymax"] - box["ymin"]
                    )
                    boxes_info.append(BoxIntersections(boxes_area, rect_start))
                    continue

        boxes_info.sort()
        # group the boxes by area
        boxes_by_area = defaultdict(list)
        for box_info in boxes_info:
            boxes_by_area[box_info.area].append(getattr(box_info, direction))

        # get midpoints for each face
        rects = [
            clamp(starts[len(starts) // 2], direction)
            for _, starts in boxes_by_area.items()
        ]

        return sorted(rects, key=lambda r: r[min_])


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
        rects = calculate_crops(image, boxes, ratio=(3, 2))
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
            rect = rects[int(chr(key)) - 1]

            cropped = image[
                rect["ymin"] : rect["ymax"], rect["xmin"] : rect["xmax"]  # noqa: E203
            ]

            cv2.imwrite(str(VERT_WALLPAPER_DIR / path.name), cropped)
            idx = idx + 1

        else:
            idx = idx + 1
