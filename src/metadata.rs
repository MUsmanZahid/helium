use std::{fs::File, io::Read};

pub(crate) fn bytes_per_pixel(color_type: u8) -> u32 {
    match color_type {
        0 => 1, // Grayscale image - one byte per pixel (G)
        2 => 3, // Truecolour image - three bytes per pixel (R, G, B)
        3 => 3, // Indexed-colour image - three bytes per pixel (R, G, B)
        4 => 2, // Grayscale with alpha - two bytes per pixel (G, A)
        6 => 4, // Truecolour with alpha - four bytes per pixel (R, G, B, A)
        _ => unreachable!(),
    }
}

pub(crate) fn contains_magic_number(f: &mut File) -> bool {
    let mut b = [0u8; 8];
    if let Err(_) = f.read_exact(&mut b[..]) {
        return false;
    }

    return b == crate::PNG_MAGIC_NUMBER;
}

pub(crate) fn header_valid(f: &mut File) -> bool {
    let mut b = [0u8; 4];
    if let Err(_) = f.read_exact(&mut b[..]) {
        return false;
    }

    // Header must have length 13.
    if u32::from_be_bytes(b) != 13 {
        return false;
    }

    if let Err(_) = f.read_exact(&mut b[..]) {
        return false;
    }
    // Chunk name must be IDAT
    return b == crate::IHDR;
}
