import cv2
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, NamedTuple, Any
import anime_face_detector


WALLPAPER_DIR = Path("~/Pictures/Wallpapers").expanduser()

VERT_WALLPAPER_DIR = Path("~/Pictures/WallpapersVertical").expanduser()
FRAMEWORK_WALLPAPER_DIR = Path("~/Pictures/WallpapersFramework").expanduser()
DETECTOR = anime_face_detector.create_detector(
    # faster-rcnn is also available
    face_detector_name="faster-rcnn",
    # "cuda:0" is also available
    device="cpu",
)

Dimensions = tuple[int, int]
AspectRatio = tuple[int, int]

HD_ASPECT_RATIO: AspectRatio = (1920, 1080)
ULTRAWIDE_ASPECT_RATIO: AspectRatio = (3440, 1440)
VERTICAL_ASPECT_RATIO: AspectRatio = (1440, 2560)
FRAMEWORK_ASPECT_RATIO: AspectRatio = (2256, 1504)
SQUARE_ASPECT_RATIO: AspectRatio = (1, 1)


class Face(TypedDict):
    xmin: int
    ymin: int
    xmax: int
    ymax: int


class FaceIntersections(NamedTuple):
    area: int
    start: int


def box_to_geometry(face: Face) -> str:
    x = face["xmin"]
    y = face["ymin"]
    w = face["xmax"] - face["xmin"]
    h = face["ymax"] - face["ymin"]

    return f"{w}x{h}+{x}+{y}"


class WallpaperInfo:
    def __init__(self):
        self.path = WALLPAPER_DIR / "wallpapers.json"
        try:
            loaded = json.load(open(self.path))
        except FileNotFoundError:
            loaded = {}

        # cleanup files that have been deleted
        self.data = {}
        for img in iter_images(WALLPAPER_DIR):
            fname = img.name

            if fname in loaded:
                self.data[fname] = loaded[fname]

    def __getitem__(self, key):
        if key not in self.data:
            return self.data[key.replace(".jpg", ".png")]
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __contains__(self, key):
        return key in self.data

    def save(self):
        json.dump(self.data, open(self.path, "w"), indent=2)


@dataclass
class Cropper:
    image: Any
    faces: list[Face]
    aspect_ratio: AspectRatio = (9, 16)

    def __init__(
        self,
        image,
        faces: list[Face],
        aspect_ratio: AspectRatio = (9, 16),
    ):
        self.image = image
        self.faces = faces
        self.height, self.width = self.image.shape[:2]
        self.set_aspect_ratio(aspect_ratio)

    def set_aspect_ratio(self, aspect_ratio: AspectRatio):
        self.aspect_ratio = aspect_ratio
        (self.target_width, self.target_height), self.direction = self.crop_rect()

    def crop_rect(self) -> tuple[Dimensions, str]:
        target_w, target_h = self.aspect_ratio

        # Calculate width and height that can be cropped while maintaining aspect ratio
        crop_w = min(self.width, int(self.height * target_w / target_h))
        crop_h = min(self.height, int(self.width * target_h / target_w))

        # Choose the larger dimension to get the largest possible cropped rectangle
        if crop_w * target_h > crop_h * target_w:
            ret = crop_w, crop_h
        else:
            ret = (
                int(crop_h * target_w / target_h),
                crop_h,
            )
        return (ret, "y" if ret[0] == self.width else "x")

    def clamp(self, val):
        min_ = int(val)
        empty = {
            "xmin": 0,
            "xmax": 0,
            "ymin": 0,
            "ymax": 0,
        }

        # check if out of bounds and constrain it
        if self.direction == "x":
            max_ = min_ + self.target_width
            if min_ < 0:
                return {**empty, "xmax": self.target_width, "ymax": self.height}
            elif max_ > self.width:
                return {
                    **empty,
                    "xmin": self.width - self.target_width,
                    "xmax": self.width,
                    "ymax": self.height,
                }
            else:
                return {**empty, "xmin": min_, "xmax": max_, "ymax": self.height}
        else:
            max_ = min_ + self.target_height
            if min_ < 0:
                return {**empty, "ymax": self.target_height, "xmax": self.width}
            elif max_ > self.height:
                return {
                    **empty,
                    "ymin": self.height - self.target_height,
                    "xmax": self.width,
                    "ymax": self.height,
                }
            else:
                return {**empty, "ymin": min_, "ymax": max_, "xmax": self.width}

    def crop_single_face(self) -> tuple[Face, list[Face]]:
        face = self.faces[0]
        face_mid = (face[f"{self.direction}min"] + face[f"{self.direction}max"]) / 2
        target = (
            face_mid - self.target_width / 2
            if self.direction == "x"
            else face_mid - self.target_height / 2
        )

        return self.clamp(target)

    def iter_image_slices(self):
        for rect_start in range(
            self.width - self.target_width
            if self.direction == "x"
            else self.height - self.target_height
        ):
            rect_end = rect_start + (
                self.target_width if self.direction == "x" else self.target_height
            )
            yield rect_start, rect_end

    def crop(self) -> Face:
        # crop area is the entire image
        if self.width == self.target_width and self.height == self.target_height:
            return {
                "xmin": 0,
                "xmax": self.width,
                "ymin": 0,
                "ymax": self.height,
            }

        if not self.faces:
            # return the center of the image
            if self.direction == "x":
                xmin = (self.width - self.target_width) // 2
                return {
                    "xmin": xmin,
                    "xmax": xmin + self.target_width,
                    "ymin": 0,
                    "ymax": self.height,
                }
            else:
                ymin = (self.height - self.target_height) // 2
                return {
                    "xmin": 0,
                    "xmax": self.width,
                    "ymin": ymin,
                    "ymax": ymin + self.target_height,
                }

        if len(self.faces) == 1:
            return self.crop_single_face()

        min_ = "xmin" if self.direction == "x" else "ymin"
        max_ = "xmax" if self.direction == "x" else "ymax"

        # sort faces by min_
        faces = sorted(self.faces, key=lambda f: f[min_])

        max_faces = 0
        # (area, min_ of face)
        faces_info: list[FaceIntersections] = []

        for rect_start, rect_end in self.iter_image_slices():
            # check number of faces in decimal within enclosed within larger rectangle
            num_faces = 0
            faces_area = 0
            for face in faces:
                # no intersection, we overshot the final face
                if face[min_] > rect_end:
                    break

                # no intersection
                elif face[max_] < rect_start:
                    continue

                # full intersection
                elif face[min_] >= rect_start and face[max_] <= rect_end:
                    num_faces += 1
                    faces_area += (face["xmax"] - face["xmin"]) * (
                        face["ymax"] - face["ymin"]
                    )
                    continue

                # partial intersection
                if face[min_] <= rect_end and face[max_] > rect_end:
                    num_faces += (rect_end - face[min_]) / (face[max_] - face[min_])
                    if self.direction == "x":
                        faces_area += (rect_end - face[min_]) * (
                            face["ymax"] - face["ymin"]
                        )
                    else:
                        faces_area += (rect_end - face[min_]) * (
                            face["xmax"] - face["xmin"]
                        )
                    continue

            # update max faces
            if num_faces > 0:
                if num_faces > max_faces:
                    max_faces = num_faces
                    faces_info = [FaceIntersections(faces_area, rect_start)]
                elif num_faces == max_faces:
                    faces_info.append(FaceIntersections(faces_area, rect_start))

        faces_info.sort()
        # use the match with the maximum area of face coverage
        max_face_area = faces_info[-1].area
        faces_info = [face for face in faces_info if face.area == max_face_area]

        # get the midpoint of matches to center the face
        return self.clamp(faces_info[len(faces_info) // 2].start)

    def crop_candidates(self) -> list[Face]:
        if len(self.faces) == 1:
            return [self.crop_single_face()]

        min_ = "xmin" if self.direction == "x" else "ymin"
        max_ = "xmax" if self.direction == "x" else "ymax"

        # sort faces by min_
        faces = sorted(self.faces, key=lambda f: f[min_])

        # (area, xmin of face)
        faces_info: list[FaceIntersections] = []

        for rect_start, rect_end in self.iter_image_slices():
            # check number of faces in decimal within enclosed within larger rectangle
            for face in faces:
                # no intersection, we overshot the final face
                if face[min_] > rect_end:
                    break

                # no intersection
                elif face[max_] < rect_start:
                    continue

                # full intersection
                elif face[min_] >= rect_start and face[max_] <= rect_end:
                    faces_area = (face["xmax"] - face["xmin"]) * (
                        face["ymax"] - face["ymin"]
                    )
                    faces_info.append(FaceIntersections(faces_area, rect_start))
                    continue

        faces_info.sort()
        # group the faces by area
        faces_by_area = defaultdict(list)
        for face_info in faces_info:
            faces_by_area[face_info.area].append(face_info.start)

        return sorted(
            # get midpoints for each face
            [
                self.clamp(starts[len(starts) // 2])
                for _, starts in faces_by_area.items()
            ],
            key=lambda r: r[min_],
        )

    def geometries(self):
        ret = {
            "faces": [(f["xmin"], f["xmax"], f["ymin"], f["ymax"]) for f in self.faces]
        }

        for ratio in [
            VERTICAL_ASPECT_RATIO,
            FRAMEWORK_ASPECT_RATIO,
            ULTRAWIDE_ASPECT_RATIO,
            HD_ASPECT_RATIO,
            SQUARE_ASPECT_RATIO,
        ]:
            ratio_str = f"{ratio[0]}x{ratio[1]}"

            self.set_aspect_ratio(ratio)
            box = self.crop()
            ret[ratio_str] = box_to_geometry(box)

        return ret


def draw(image, faces, color=(0, 255, 0), thickness=1):
    """
    Draw boxes on the image. This function does not modify the image in-place.
    Args:
        image: A numpy BGR image.
        faces: A list of dicts of {xmin, xmax, ymin, ymax}
        colors: Color (BGR) used to draw.
        thickness: Thickness of the line.
    Returns:
        A drawn image.
    """
    image = image.copy()
    for face in faces:
        xmin, ymin, xmax, ymax = face["xmin"], face["ymin"], face["xmax"], face["ymax"]
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)
    return image


def detect(
    img,
    face_score_threshold: float,
) -> list[Face]:
    image = cv2.imread(img)
    preds = DETECTOR(image)

    faces = []
    for pred in preds:
        face = pred["bbox"]
        score = face[4]
        if score < face_score_threshold:
            continue

        faces.append(
            {
                "xmin": int(face[0]),
                "ymin": int(face[1]),
                "xmax": int(face[2]),
                "ymax": int(face[3]),
            }
        )

    return faces


def iter_images(p: Path):
    for img in p.iterdir():
        if not img.is_file():
            continue

        if img.suffix == ".json":
            continue

        yield img
