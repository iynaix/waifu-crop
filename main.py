import pathlib

import cv2
import numpy as np
import PIL.Image
import torch
import anime_face_detector


def detect(
    img,
    face_score_threshold: float,
    landmark_score_threshold: float,
    detector: anime_face_detector.LandmarkDetector,
) -> PIL.Image.Image:
    image = cv2.imread(img.name)
    preds = detector(image)

    res = image.copy()
    for pred in preds:
        box = pred["bbox"]
        box, score = box[:4], box[4]
        if score < face_score_threshold:
            continue
        box = np.round(box).astype(int)

        lt = max(2, int(3 * (box[2:] - box[:2]).max() / 256))

        cv2.rectangle(res, tuple(box[:2]), tuple(box[2:]), (0, 255, 0), lt)

        pred_pts = pred["keypoints"]
        for *pt, score in pred_pts:
            if score < landmark_score_threshold:
                color = (0, 255, 255)
            else:
                color = (0, 0, 255)
            pt = np.round(pt).astype(int)
            cv2.circle(res, tuple(pt), lt, color, cv2.FILLED)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)

    image_pil = PIL.Image.fromarray(res)
    return image_pil


if __name__ == "__main__":
    sample_path = pathlib.Path("/home/iynaix/Pictures/Wallpapers/wallhaven-85dw1j.jpg")

    # faster-rcnn is also available
    detector_model = "yolov3"
    # cpu is also available
    # detector_device = "cuda:0"
    detector_device = "cpu"

    detector = anime_face_detector.create_detector(
        face_detector_name=detector_model,
        device=detector_device,
    )

    detect(
        sample_path.as_posix(),
        face_score_threshold=0.5,
        landmark_score_threshold=0.3,
        detector=detector,
    )
