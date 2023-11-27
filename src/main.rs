use iced::widget::{column, container, image, text};
use iced::{Element, Sandbox, Settings, Theme};
use itertools::Itertools;

use wallpaper_utils::{wallpaper_dir, Cropper, WallInfo, VERTICAL_ASPECT_RATIO};

struct Counter {
    // The counter value
    image: image::Handle,
    info: WallInfo,
}

#[derive(Debug, Clone, Copy)]
pub enum Message {
    // IncrementPressed,
    // DecrementPressed,
}

impl Sandbox for Counter {
    type Message = Message;

    fn new() -> Self {
        let image_data = wallpaper_utils::read_wallpaper_info();

        let (fname, wall_info) = image_data
            .into_iter()
            .sorted_by_key(|(fname, _)| fname.to_string())
            .find_map(|(fname, info)| match info {
                info if info.faces.len() > 1 => Some((fname, info)),
                _ => None,
            })
            .unwrap();

        // {
        //     if let Some(info) = info {
        //         if info.faces.len() <= 1 {
        //             continue;
        //         }

        //         println!("{}", fname);

        //         let mut cropper = Cropper::new(fname.to_string(), info.faces);

        //         let candidates = cropper.crop_candidates(VERTICAL_ASPECT_RATIO);
        //         println!("{:?}", &candidates);
        //     }
        // }

        Self {
            image: image::Handle::from_path(wallpaper_dir().join(fname)),
            info: wall_info,
        }
    }

    fn title(&self) -> String {
        String::from("Counter - Iced")
    }

    fn theme(&self) -> Theme {
        Theme::Dark
    }

    fn update(&mut self, message: Message) {
        match message {
            // Message::IncrementPressed => {
            //     self.value += 1;
            // }
            // Message::DecrementPressed => {
            //     self.value -= 1;
            // }
        }
    }

    fn view(&self) -> Element<Message> {
        // column![
        //     button("Increment").on_press(Message::IncrementPressed),
        //     text(self.value).size(50),
        //     button("Decrement").on_press(Message::DecrementPressed)
        // ]
        // .padding(20)
        // .align_items(Alignment::Center)
        // .into()

        column![
            container(image::viewer(self.image.clone()).padding(10))
                .width(iced::Length::Fill)
                .height(iced::Length::Fill)
                .padding(10)
                .center_x()
                .center_y(),
            text(format!("{:?}", self.info.faces))
        ]
        .into()
    }
}

fn main() -> iced::Result {
    Counter::run(Settings::default())
}
