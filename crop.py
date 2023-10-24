import cv2
from typing import TypedDict, NamedTuple
from pathlib import Path


WALLPAPER_DIR = Path("~/Pictures/Wallpapers").expanduser()
VERT_WALLPAPER_DIR = Path("~/Pictures/WallpapersVertical").expanduser()


class Box(TypedDict):
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    confidence: float


class BoxIntersections(NamedTuple):
    area: int
    x: int


def calculate_crop(image, boxes: list[Box]) -> tuple[Box, list[Box]]:
    height, width = image.shape[:2]
    target_width = int(height / 16 * 9)

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

        max_boxes = 0
        # (area, xmin of box)
        boxes_info: list[BoxIntersections] = []

        for rect_left in range(width - target_width):
            rect_right = rect_left + target_width

            # check number of boxes in decimal within enclosed within larger rectangle
            num_boxes = 0
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
                    num_boxes += 1
                    boxes_area += (box["xmax"] - box["xmin"]) * (
                        box["ymax"] - box["ymin"]
                    )
                    continue

                # partial intersection
                if box["xmin"] <= rect_right and box["xmax"] > rect_right:
                    num_boxes += (rect_right - box["xmin"]) / (
                        box["xmax"] - box["xmin"]
                    )
                    boxes_area += (rect_right - box["xmin"]) * (
                        box["ymax"] - box["ymin"]
                    )
                    continue

            # update max boxes
            if num_boxes > 0:
                if num_boxes > max_boxes:
                    max_boxes = num_boxes
                    boxes_info = [BoxIntersections(boxes_area, rect_left)]
                elif num_boxes == max_boxes:
                    boxes_info.append(BoxIntersections(boxes_area, rect_left))

        boxes_info.sort()
        # use the match with the maximum area of face coverage
        max_box_area = boxes_info[-1].area
        boxes_info = [box for box in boxes_info if box.area == max_box_area]

        # get the midpoint of matches to center the box
        start_x = boxes_info[len(boxes_info) // 2].x
        xmin, xmax = clamp(start_x)

        return (
            {
                "xmin": xmin,
                "xmax": xmax,
                "ymin": 0,
                "ymax": height,
                "confidence": boxes[0]["confidence"],
            },
            boxes,
        )


def write_cropped_image(image, boxes: list[Box], filename: str):
    rect, _ = calculate_crop(image, boxes)

    cropped = image[
        rect["ymin"] : rect["ymax"], rect["xmin"] : rect["xmax"]  # noqa: E203
    ]

    cv2.imwrite(str(VERT_WALLPAPER_DIR / filename), cropped)
