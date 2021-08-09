use std::{
    error::Error,
    fs::File,
    io::{BufWriter, Write},
};

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args();
    let process_name = match args.next() {
        Some(name) => name,
        None => String::from("????"),
    };
    let mut file_name = match args.next() {
        Some(name) => name,
        None => {
            eprintln!("Usage: {} <filename>", process_name);
            return Ok(());
        }
    };
    let write_ppm = match args.next() {
        Some(i) => i == "true",
        None => false,
    };

    let image = helium::helium(&file_name)?;
    if write_ppm {
        file_name += ".ppm";

        let file = File::create(file_name)?;
        let mut writer = BufWriter::new(file);
        let header = format!("P6\n{} {}\n255\n", image.width, image.height);
        writer.write_all(header.as_bytes())?;

        if image.num_channels == 3 {
            writer.write_all(&image.data)?;
        } else if image.num_channels == 4 {
            for pixel in 0..(image.width * image.height) as usize {
                let pixel_index = 4 * pixel;
                writer.write_all(&image.data[pixel_index..(pixel_index + 3)])?;
            }
        }
    }

    Ok(())
}
