use itertools::Itertools;
use wallpaper_utils::{Cropper, VERTICAL_ASPECT_RATIO};

fn main() {
    let image_data = wallpaper_utils::read_wallpaper_info();

    for (fname, info) in image_data
        .into_iter()
        .sorted_by_key(|(fname, _)| fname.to_string())
    {
        if let Some(info) = info {
            if info.faces.len() <= 1 {
                continue;
            }

            println!("{}", fname);

            let mut cropper = Cropper::new(fname.to_string(), info.faces);

            let candidates = cropper.crop_candidates(VERTICAL_ASPECT_RATIO);
            println!("{:?}", &candidates);
        }
    }
}
