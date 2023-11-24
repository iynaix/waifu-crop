use itertools::Itertools;
use wallpaper_utils::{Cropper, FRAMEWORK_ASPECT_RATIO, HD_ASPECT_RATIO, ULTRAWIDE_ASPECT_RATIO};

fn main() {
    let image_data = wallpaper_utils::read_wallpaper_info();

    for (fname, info) in image_data
        .into_iter()
        .sorted_by_key(|(fname, _)| fname.to_string())
    {
        if let Some(info) = info {
            println!("{}", fname);

            let mut cropper = Cropper::new(fname.to_string(), info.faces);

            assert_eq!(
                info.r3440x1440,
                cropper.crop(ULTRAWIDE_ASPECT_RATIO).geometry(),
                "ULTRAWIDE_ASPECT_RATIO"
            );

            assert_eq!(
                info.r2256x1504,
                cropper.crop(FRAMEWORK_ASPECT_RATIO).geometry(),
                "FRAMEWORK_ASPECT_RATIO"
            );

            assert_eq!(
                info.r1920x1080,
                cropper.crop(HD_ASPECT_RATIO).geometry(),
                "HD_ASPECT_RATIO"
            );
        }
    }
}
