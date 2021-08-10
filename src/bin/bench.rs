fn main() -> Result<(), std::io::Error> {
    let files: Vec<std::path::PathBuf> = std::fs::read_dir("Performance Suite")?
        .filter_map(|entry| {
            if let Ok(entry) = entry {
                if entry.path().is_file() {
                    return Some(entry.path());
                }
            }

            None
        })
        .collect();

    files.into_iter().for_each(|file| {
        let file_name = file.as_os_str().to_str().unwrap();
        println!("{}", file_name);
        helium::helium(file_name).unwrap();
        println!();
    });

    Ok(())
}
