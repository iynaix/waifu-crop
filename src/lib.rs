use core::fmt;
use image::image_dimensions;
use itertools::Itertools;
use serde::Deserialize;
use std::{collections::HashMap, fs::File, path::PathBuf};

pub struct AspectRatio(u32, u32);

pub const HD_ASPECT_RATIO: AspectRatio = AspectRatio(1920, 1080);
pub const ULTRAWIDE_ASPECT_RATIO: AspectRatio = AspectRatio(3440, 1440);
pub const VERTICAL_ASPECT_RATIO: AspectRatio = AspectRatio(1440, 2560);
pub const FRAMEWORK_ASPECT_RATIO: AspectRatio = AspectRatio(2256, 1504);

impl fmt::Display for AspectRatio {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}x{}", self.0, self.1)
    }
}

pub fn full_path(p: &str) -> PathBuf {
    match p.strip_prefix("~/") {
        Some(p) => dirs::home_dir().unwrap().join(p),
        None => PathBuf::from(p),
    }
}

pub fn wallpaper_dir() -> PathBuf {
    full_path("~/Pictures/Wallpapers")
}

#[derive(Debug, Default, Deserialize)]
pub struct Face {
    #[serde(rename = "0")]
    pub xmin: u32,
    #[serde(rename = "1")]
    pub xmax: u32,
    #[serde(rename = "2")]
    pub ymin: u32,
    #[serde(rename = "3")]
    pub ymax: u32,
}

impl Face {
    pub fn width(&self) -> u32 {
        self.xmax - self.xmin
    }

    pub fn height(&self) -> u32 {
        self.ymax - self.ymin
    }

    pub fn area(&self) -> u32 {
        (self.xmax - self.xmin) * (self.ymax - self.ymin)
    }

    pub fn geometry(&self) -> String {
        format!(
            "{}x{}+{}+{}",
            self.width(),
            self.height(),
            self.xmin,
            self.ymin
        )
    }
}

type Rect = Face;

struct FaceInfo {
    area: u32,
    start: u32,
}

type WallpapersInfo = HashMap<String, WallInfo>;

#[derive(Debug, Deserialize)]
pub struct WallInfo {
    pub filter: String,
    pub faces: Vec<Face>,
    #[serde(rename = "1440x2560")]
    pub r1440x2560: String,
    #[serde(rename = "2256x1504")]
    pub r2256x1504: String,
    #[serde(rename = "3440x1440")]
    pub r3440x1440: String,
    #[serde(rename = "1920x1080")]
    pub r1920x1080: String,
}

pub struct Cropper {
    pub faces: Vec<Face>,
    pub image: PathBuf,
    pub width: u32,
    pub height: u32,
    pub aspect_ratio: AspectRatio,
}

#[derive(Debug, Clone)]
pub enum Direction {
    X,
    Y,
}

impl Cropper {
    pub fn new(image: String, faces: Vec<Face>) -> Self {
        let image = wallpaper_dir().join(image);
        let (width, height) = image_dimensions(&image).expect("Failed to read image dimensions");

        Self {
            faces,
            image,
            width,
            height,
            aspect_ratio: VERTICAL_ASPECT_RATIO,
        }
    }

    pub fn set_aspect_ratio(&mut self, aspect_ratio: AspectRatio) {
        self.aspect_ratio = aspect_ratio;
    }

    pub fn crop_rect(&self, aspect_ratio: AspectRatio) -> (u32, u32, Direction) {
        let AspectRatio(target_w, target_h) = aspect_ratio;

        // Calculate width and height that can be cropped while maintaining aspect ratio
        let crop_w = std::cmp::min(self.width, self.height * target_w / target_h);
        let crop_h = std::cmp::min(self.height, self.width * target_h / target_w);

        // Choose the larger dimension to get the largest possible cropped rectangle
        let (crop_w, crop_h) = if crop_w * target_h > crop_h * target_w {
            (crop_w, crop_h)
        } else {
            (crop_h * target_w / target_h, crop_h)
        };

        (
            crop_w,
            crop_h,
            if crop_w == self.width {
                Direction::Y
            } else {
                Direction::X
            },
        )
    }

    pub fn clamp(
        &self,
        val: f32,
        direction: Direction,
        target_width: u32,
        target_height: u32,
    ) -> Face {
        let min_ = val;
        match direction {
            Direction::X => {
                let max_ = min_ + target_width as f32;
                if min_ < 0.0 {
                    Face {
                        xmax: target_width,
                        ymax: self.height,
                        ..Face::default()
                    }
                } else if max_ > self.width as f32 {
                    Face {
                        xmin: self.width - target_width,
                        xmax: self.width,
                        ymax: self.height,
                        ..Face::default()
                    }
                } else {
                    Face {
                        xmin: min_ as u32,
                        xmax: max_ as u32,
                        ymax: self.height,
                        ..Face::default()
                    }
                }
            }
            Direction::Y => {
                let max_ = min_ + target_height as f32;
                if min_ < 0.0 {
                    Face {
                        ymax: target_height,
                        xmax: self.width,
                        ..Face::default()
                    }
                } else if max_ > self.height as f32 {
                    Face {
                        ymin: self.height - target_height,
                        ymax: self.height,
                        xmax: self.width,
                        ..Face::default()
                    }
                } else {
                    Face {
                        ymin: min_ as u32,
                        ymax: max_ as u32,
                        xmax: self.width,
                        ..Face::default()
                    }
                }
            }
        }
    }

    pub fn crop_single_face(
        &self,
        direction: Direction,
        target_width: u32,
        target_height: u32,
    ) -> Rect {
        let Face {
            xmin,
            xmax,
            ymin,
            ymax,
        } = &self.faces[0];
        match direction {
            Direction::X => {
                let mid = (xmin + xmax) as f32 / 2.0;
                self.clamp(
                    mid - (target_width as f32 / 2.0),
                    direction,
                    target_width,
                    target_height,
                )
            }
            Direction::Y => {
                let mid = (ymin + ymax) as f32 / 2.0;
                self.clamp(
                    mid - (target_height as f32 / 2.0),
                    direction,
                    target_width,
                    target_height,
                )
            }
        }
    }

    pub fn crop(&mut self, aspect_ratio: AspectRatio) -> Rect {
        let (target_width, target_height, direction) = self.crop_rect(aspect_ratio);

        if self.width == target_width && self.height == target_height {
            return Face {
                xmax: self.width,
                ymax: self.height,
                ..Face::default()
            };
        }

        if self.faces.len() == 1 {
            self.crop_single_face(direction, target_width, target_height)
        } else {
            self.faces.sort_by_key(|face| match direction {
                Direction::X => face.xmin,
                Direction::Y => face.ymin,
            });

            let mut max_faces = 0.0;
            let mut faces_info: Vec<FaceInfo> = vec![];
            let (rect_max, rect_len) = match direction {
                Direction::X => (self.width - target_width, target_width),
                Direction::Y => (self.height - target_height, target_height),
            };

            for rect_start in 0..rect_max {
                let rect_end = rect_start + rect_len;
                let mut num_faces: f32 = 0.0;
                let mut faces_area = 0;

                for face in self.faces.iter() {
                    // check number of faces in decimal within enclosed within larger rectangle
                    let (min_, max_) = match direction {
                        Direction::X => (face.xmin, face.xmax),
                        Direction::Y => (face.ymin, face.ymax),
                    };

                    // no intersection, we overshot the final box
                    if min_ > rect_end {
                        break;
                    }
                    // no intersection
                    else if max_ < rect_start {
                        continue;
                    }
                    // full intersection
                    else if min_ >= rect_start && max_ <= rect_end {
                        num_faces += 1.0;
                        faces_area += face.area();
                        continue;
                    }

                    // partial intersection
                    if min_ <= rect_end && max_ > rect_end {
                        num_faces += (rect_end - min_) as f32 / (max_ - min_) as f32;
                        faces_area += (rect_end - min_)
                            * match direction {
                                Direction::X => face.height(),
                                Direction::Y => face.width(),
                            };
                        continue;
                    }
                }

                // update max_boxes
                if num_faces > 0.0 {
                    if num_faces > max_faces {
                        max_faces = num_faces;
                        faces_info = vec![FaceInfo {
                            area: faces_area,
                            start: rect_start,
                        }];
                    } else if num_faces == max_faces {
                        faces_info.push(FaceInfo {
                            area: faces_area,
                            start: rect_start,
                        });
                    }
                }
            }

            faces_info.sort_by_key(|face_info| (face_info.area, face_info.start));
            // use the match with the maximum area of face coverage
            let max_face_area = faces_info.last().unwrap().area;
            faces_info.retain(|face_info| face_info.area == max_face_area);

            self.clamp(
                faces_info[faces_info.len() / 2].start as f32,
                direction,
                target_width,
                target_height,
            )
        }
    }

    pub fn crop_candidates(&mut self, aspect_ratio: AspectRatio) -> Vec<Rect> {
        let (target_width, target_height, direction) = self.crop_rect(aspect_ratio);

        if self.width == target_width && self.height == target_height {
            return vec![Face {
                xmax: self.width,
                ymax: self.height,
                ..Face::default()
            }];
        }

        if self.faces.len() == 1 {
            vec![self.crop_single_face(direction, target_width, target_height)]
        } else {
            self.faces.sort_by_key(|face| match direction {
                Direction::X => face.xmin,
                Direction::Y => face.ymin,
            });

            let mut faces_info: Vec<FaceInfo> = vec![];
            let (rect_max, rect_len) = match direction {
                Direction::X => (self.width - target_width, target_width),
                Direction::Y => (self.height - target_height, target_height),
            };

            for rect_start in 0..rect_max {
                // check number of faces in decimal within enclosed within larger rectangle
                let rect_end = rect_start + rect_len;

                for face in self.faces.iter() {
                    let (min_, max_) = match direction {
                        Direction::X => (face.xmin, face.xmax),
                        Direction::Y => (face.ymin, face.ymax),
                    };

                    // no intersection, we overshot the final box
                    if min_ > rect_end {
                        break;
                    }
                    // no intersection
                    else if max_ < rect_start {
                        continue;
                    }
                    // full intersection
                    else if min_ >= rect_start && max_ <= rect_end {
                        faces_info.push(FaceInfo {
                            area: face.area(),
                            start: rect_start,
                        });
                        continue;
                    }
                }
            }

            faces_info.sort_by_key(|face_info| (face_info.area, face_info.start));

            // group faces by area
            let faces_by_area: HashMap<_, Vec<_>> =
                faces_info
                    .iter()
                    .fold(HashMap::new(), |mut acc, face_info| {
                        acc.entry(face_info.area).or_default().push(face_info.start);
                        acc
                    });

            faces_by_area
                .values()
                .map(|faces| {
                    let mid = faces[faces.len() / 2];
                    self.clamp(mid as f32, direction.clone(), target_width, target_height)
                })
                .sorted_by_key(|face| match direction {
                    Direction::X => face.xmin,
                    Direction::Y => face.ymin,
                })
                .collect()
        }
    }
}

// read ~/Pictures/Wallpapers/wallpaper.json
pub fn read_wallpaper_info() -> WallpapersInfo {
    let path = wallpaper_dir().join("wallpapers.json");
    let reader = std::io::BufReader::new(File::open(path).unwrap());
    serde_json::from_reader(reader).unwrap()
}
