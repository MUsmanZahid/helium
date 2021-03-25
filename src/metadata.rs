use std::{fs::File, io::Read};
use crate::{read_u8, read_u32, MetadataError, HeliumImageMetadata};

// TODO: Don't panic here!
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

pub(crate) fn get(file_name: &str) -> Result<HeliumImageMetadata, MetadataError> {
    let mut file = File::open(file_name)?;

    if !contains_magic_number(&mut file) {
        return Err(MetadataError::InvalidMagicNumber);
    }
    if !header_valid(&mut file) {
        return Err(MetadataError::InvalidHeader);
    }

    let width = read_u32(&mut file)?;
    let height = read_u32(&mut file)?;
    let bit_depth = read_u8(&mut file)?;
    let color_type = read_u8(&mut file)?;
    let bits_per_pixel = bit_depth as u32 * bytes_per_pixel(color_type);

    let metadata = HeliumImageMetadata {
        width,
        height,
        bits_per_pixel,
    };
    return Ok(metadata);
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
