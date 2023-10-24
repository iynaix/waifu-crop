import cv2
import anime_face_detector
from crop import WALLPAPER_DIR, VERT_WALLPAPER_DIR, write_cropped_image, calculate_crop


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
    # faster-rcnn is also available
    detector_model = "yolov3"
    # cpu is also available
    # detector_device = "cuda:0"
    detector_device = "cpu"

    detector = anime_face_detector.create_detector(
        face_detector_name=detector_model,
        device=detector_device,
    )

    # skip images if already cropped
    vertical_wallpapers = set(f.name for f in VERT_WALLPAPER_DIR.iterdir())
    image_paths = [
        img for img in WALLPAPER_DIR.iterdir() if img.name not in vertical_wallpapers
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

    for path in image_paths:
        # use defaults
        image = cv2.imread(str(path))
        boxes = detect(
            str(path),
            face_score_threshold=0.5,
            detector=detector,
        )

        # skip if no boxes
        if not boxes:
            continue

        print(path, "x".join(image.shape[:2:-1]))

        # display the images
        if PREVIEW_IMAGES:
            rect, detection_boxes = calculate_crop(image, boxes)
            boxes_to_draw = [rect, *detection_boxes]

            drawn_image = draw(
                image,
                boxes_to_draw,
                # BGR
                color=(0, 255, 0),
                thickness=3,
            )

            w, h = image.shape[:2][::-1]
            resized_image = cv2.resize(drawn_image, (int(w / h * 720), 720))
            cv2.imshow("Image", resized_image)

            key = cv2.waitKey(0) & 0xFF
            # esc
            if key == ord("q") or key == 27:
                break
            # right arrow
            elif key == ord("n") or key == 39:
                continue
        # write to file
        else:
            write_cropped_image(image, boxes, path.name)
