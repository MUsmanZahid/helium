fn main() {
    let mut args = std::env::args();
    let process_name = match args.next() {
        Some(name) => name,
        None => String::from("????"),
    };
    let file_name = match args.next() {
        Some(name) => name,
        None => {
            eprintln!("Usage: {} <filename>", process_name);
            return;
        }
    };

    match helium::helium(&file_name) {
        Ok(_) => (),
        Err(e) => eprintln!("{}", e),
    }
}
