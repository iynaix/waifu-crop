import cv2
from typing import TypedDict, NamedTuple
from pathlib import Path
import anime_face_detector


WALLPAPER_DIR = Path("~/Pictures/Wallpapers").expanduser()
VERT_WALLPAPER_DIR = Path("~/Pictures/WallpapersVertical").expanduser()
DETECTOR = anime_face_detector.create_detector(
    # faster-rcnn is also available
    face_detector_name="faster-rcnn",
    # "cuda:0" is also available
    device="cpu",
)

Dimensions = tuple[int, int]
AspectRatio = tuple[int, int]


class Box(TypedDict):
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    confidence: float


class BoxIntersections(NamedTuple):
    area: int
    x: int


def get_largest_crop(
    image_dims: Dimensions,
    aspect_ratio: AspectRatio = (9, 16),
) -> tuple[Dimensions, str]:
    image_width, image_height = image_dims
    target_width, target_height = aspect_ratio

    # Calculate width and height that can be cropped while maintaining the aspect ratio
    crop_width = min(image_width, int(image_height * target_width / target_height))
    crop_height = min(image_height, int(image_width * target_height / target_width))

    # Choose the larger dimension to get the largest possible cropped rectangle
    if crop_width * target_height > crop_height * target_width:
        return (crop_width, crop_height), "y"
    else:
        return (int(crop_height * target_width / target_height), crop_height), "x"


def calculate_crop(
    image,
    boxes: list[Box],
    aspect_ratio: AspectRatio = (9, 16),
) -> tuple[Box, list[Box]]:
    height, width = image.shape[:2]
    (target_width, target_height), direction = get_largest_crop(
        (width, height), aspect_ratio
    )

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

        return (
            clamp(target, direction),
            boxes,
        )

    else:
        min_ = "xmin" if direction == "x" else "ymin"
        max_ = "xmax" if direction == "x" else "ymax"

        # sort boxes by min_
        boxes.sort(key=lambda box: box[min_])

        max_boxes = 0
        # (area, xmin of box)
        boxes_info: list[BoxIntersections] = []

        for rect_start in range(
            width - target_width if direction == "x" else height - target_height
        ):
            rect_end = rect_start + (
                target_width if direction == "x" else target_height
            )

            # check number of boxes in decimal within enclosed within larger rectangle
            num_boxes = 0
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
                    num_boxes += 1
                    boxes_area += (box["xmax"] - box["xmin"]) * (
                        box["ymax"] - box["ymin"]
                    )
                    continue

                # partial intersection
                if box[min_] <= rect_end and box[max_] > rect_end:
                    num_boxes += (rect_end - box[min_]) / (box[max_] - box[min_])
                    if direction == "x":
                        boxes_area += (rect_end - box[min_]) * (
                            box["ymax"] - box["ymin"]
                        )
                    else:
                        boxes_area += (rect_end - box[min_]) * (
                            box["xmax"] - box["xmin"]
                        )
                    continue

            # update max boxes
            if num_boxes > 0:
                if num_boxes > max_boxes:
                    max_boxes = num_boxes
                    boxes_info = [BoxIntersections(boxes_area, rect_start)]
                elif num_boxes == max_boxes:
                    boxes_info.append(BoxIntersections(boxes_area, rect_start))

        boxes_info.sort()
        # use the match with the maximum area of face coverage
        max_box_area = boxes_info[-1].area
        boxes_info = [box for box in boxes_info if box.area == max_box_area]

        # get the midpoint of matches to center the box
        start = getattr(boxes_info[len(boxes_info) // 2], direction)

        return (
            clamp(start, direction),
            boxes,
        )


def draw(image, boxes, color=(0, 255, 0), thickness=1):
    """
    Draw boxes on the image. This function does not modify the image in-place.
    Args:
        image: A numpy BGR image.
        boxes: A list of dicts of {xmin, xmax, ymin, ymax, confidence}
        colors: Color (BGR) used to draw.
        thickness: Thickness of the line.
    Returns:
        A drawn image.
    """
    image = image.copy()
    for box in boxes:
        xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)
    return image


def preview_image(
    image,
    boxes: list[Box],
    idx: int,
    # (width, height)
    ratio: tuple[int, int] = (9, 16),
) -> int:
    rect, detection_boxes = calculate_crop(image, boxes, ratio)
    boxes_to_draw = [rect, *detection_boxes]

    drawn_image = draw(
        image,
        boxes_to_draw,
        # BGR
        color=(0, 255, 0),
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


def detect(
    img,
    face_score_threshold: float,
):
    image = cv2.imread(img)
    preds = DETECTOR(image)

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
