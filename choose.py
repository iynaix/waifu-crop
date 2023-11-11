import cv2
from pathlib import Path
from collections import defaultdict
from crop import (
    Box,
    BoxIntersections,
    VERT_WALLPAPER_DIR,
    detect,
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
    ratio: tuple[int, int] = (9, 16),
) -> tuple[Box, list[Box]]:
    height, width = image.shape[:2]
    if ratio[1] > ratio[0]:
        target_width = int(height / ratio[1] * ratio[0])
    else:
        target_width = int(width / ratio[0] * ratio[1])

    def clamp(xmin) -> tuple[int, int]:
        xmin = int(xmin)
        xmax = xmin + target_width

        # check if out of bounds and constrain it
        if xmin < 0:
            return 0, target_width
        elif xmax > width:
            return width - target_width, width
        else:
            return xmin, xmax

    if len(boxes) == 1:
        box = boxes[0]
        box_mid_x = (box["xmin"] + box["xmax"]) / 2
        target_x = box_mid_x - target_width / 2

        xmin, xmax = clamp(target_x)

        return (
            {
                "xmin": xmin,
                "xmax": xmax,
                "ymin": 0,
                "ymax": height,
                "confidence": box["confidence"],
            },
            boxes,
        )

    else:
        # sort boxes by xmin
        boxes.sort(key=lambda box: box["xmin"])

        # (area, xmin of box)
        boxes_info: list[BoxIntersections] = []

        for rect_left in range(width - target_width):
            rect_right = rect_left + target_width

            # check number of boxes in decimal within enclosed within larger rectangle
            boxes_area = 0
            for box in boxes:
                # no intersection, we overshot the final box
                if box["xmin"] > rect_right:
                    break

                # no intersection
                elif box["xmax"] < rect_left:
                    continue

                # full intersection
                elif box["xmin"] >= rect_left and box["xmax"] <= rect_right:
                    boxes_area += (box["xmax"] - box["xmin"]) * (
                        box["ymax"] - box["ymin"]
                    )
                    boxes_info.append(BoxIntersections(boxes_area, rect_left))
                    continue

        boxes_info.sort()

        # group the boxes by area
        boxes_by_area = defaultdict(list)
        for box_info in boxes_info:
            boxes_by_area[box_info.area].append(box_info.x)

        # get midpoints for each face
        rects = []
        for _, xs in boxes_by_area.items():
            xmin, xmax = clamp(xs[len(xs) // 2])

            rects.append(
                {
                    "xmin": xmin,
                    "xmax": xmax,
                    "ymin": 0,
                    "ymax": height,
                    "confidence": boxes[0]["confidence"],
                }
            )

        return sorted(rects, key=lambda r: r["xmin"])


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
        rects = calculate_crops(image, boxes)
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
