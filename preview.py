import cv2
from pathlib import Path
from utils import (
    Cropper,
    # FRAMEWORK_ASPECT_RATIO,
    Face,
    VERTICAL_ASPECT_RATIO,
    WALLPAPER_DIR,
    detect,
    draw,
    iter_images,
)


def preview_image(
    image,
    faces: list[Face],
    idx: int,
    # (width, height)
    ratio: tuple[int, int] = (9, 16),
) -> int:
    rect = Cropper(image, faces, ratio).crop()

    drawn_image = draw(
        image,
        [rect, *faces],
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
    # skip images if already cropped
    image_paths = list(iter_images(WALLPAPER_DIR))
    # image_paths = list(iter_images(Path("in")))

    # uncomment to test specific images
    # image_paths = sorted(iter_images(Path("in")))

    print("Start inferencing. Press `q` to cancel. Press  `-` to go back.")
    idx = 0
    while True:
        if idx >= len(image_paths) or idx < 0:
            break

        path = image_paths[idx]

        # use defaults
        image = cv2.imread(str(path))
        faces = detect(
            str(path),
            face_score_threshold=0.5,
        )

        # skip if no faces
        if not faces:
            idx += 1
            continue

        # display the images
        idx = preview_image(
            image,
            faces,
            idx,
            ratio=VERTICAL_ASPECT_RATIO,
            # ratio=FRAMEWORK_ASPECT_RATIO,
        )
