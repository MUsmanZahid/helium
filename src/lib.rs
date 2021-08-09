#![allow(clippy::not_unsafe_ptr_arg_deref)]
mod metadata;

#[global_allocator]
static __: ::std::alloc::System = ::std::alloc::System;

use std::{
    error::Error,
    ffi::c_void,
    fmt::{self, Display, Formatter},
    fs::File,
    io::Read,
    mem::ManuallyDrop,
    slice,
};

const BUFFER_SIZE: usize = 4 * 1024;
const DISTANCES: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const DISTANCE_EXTRA_BITS: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];
const IDAT: [u8; 4] = [73, 68, 65, 84];
const IEND: [u8; 4] = [73, 69, 78, 68];
const IHDR: [u8; 4] = [73, 72, 68, 82];
const LENGTHS: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
const LENGTH_EXTRA_BITS: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
const PNG_MAGIC_NUMBER: [u8; 8] = [137, 80, 78, 71, 13, 10, 26, 10];
const NUM_8BIT_CRCS: usize = 256;

struct BitBuffer<'s> {
    bits: u64,
    bit_count: u8,
    index: usize,
    length: usize,
    stream: &'s [u8],
}

impl<'s> BitBuffer<'s> {
    /// Consumes count number of bits from the stream stored in the BitBuffer.
    /// Returns a 16-bit unsigned integer unless the stream has overflowed in which case it returns
    /// an error.
    fn bits(&mut self, count: u8) -> Result<u16, ZlibError> {
        let value = self.peek_bits(count)?;
        self.throw_bits(count);
        Ok(value as u16)
    }

    /// Retrieves count number of bits from the stream stored in the BitBuffer without consuming
    /// them.
    ///
    /// Returns a 16-bit unsigned integer unless the stream has overflowed, in which case it
    /// returns an error.
    fn peek_bits(&mut self, count: u8) -> Result<u16, ZlibError> {
        // We try to store as many bits as we can at a given time in order to more successfully
        // predict this branch.
        if self.bit_count < count {
            let repeat = usize::min((64 - self.bit_count as usize) / 8, self.length - self.index);
            for _ in 0..repeat {
                match self.stream.get(self.index) {
                    Some(&i) => self.bits |= (i as u64) << self.bit_count,
                    None => return Err(ZlibError::StreamOverflow),
                }
                self.bit_count += 8;
                self.index += 1;
            }
        }

        let value = self.bits & ((1u64 << count) - 1);
        Ok(value as u16)
    }

    /// Discards count bits from the BitBuffer.
    fn throw_bits(&mut self, count: u8) {
        self.bits >>= count;
        self.bit_count -= count;
    }
}

struct Header {
    width: usize,
    height: usize,
    bytes_per_pixel: usize,
}

impl Header {
    fn parse(crc_table: &[u32; 256], file: &mut File) -> Result<Self, PngError> {
        // First four bytes are the length and next four are the name of the header chunk. The length
        // must be 13 and the name must be a four-byte string "IHDR".
        if !metadata::header_valid(file) {
            return Err(PngError::InvalidHeader);
        }

        // NOTE: We do a manual read here as we need the bytes in a contiguous format for the
        // `update_crc` function we use later on!
        let mut b = [0; 17];
        b[0] = b'I';
        b[1] = b'H';
        b[2] = b'D';
        b[3] = b'R';
        file.read_exact(&mut b[4..17])?;

        // The fields of the IHDR chunk are ordered as retrieved below
        let width = u32::from_be_bytes([b[4], b[5], b[6], b[7]]) as usize;
        let height = u32::from_be_bytes([b[8], b[9], b[10], b[11]]) as usize;
        let bit_depth = b[12];
        let colour_type = b[13];
        let compression_method = b[14];
        let filter_method = b[15];
        let interlace_method = b[16];

        // The interlace method can be 0 or 1. We only support non-interlaced images for now. The filter
        // and compression methods must both be zero. Finally, we only support images with bit-depths of
        // 8 bits-per-pixel.
        if interlace_method != 0 {
            return Err(PngError::UnsupportedInterlaceMethod);
        } else if filter_method != 0 {
            return Err(PngError::NonZeroFilterMethod);
        } else if compression_method != 0 {
            return Err(PngError::NonZeroCompressionMethod);
        } else if bit_depth != 8 {
            return Err(PngError::UnsupportedBitDepth(bit_depth));
        }

        let crc = update_crc(0xFFFFFFFF, &b, crc_table) ^ 0xFFFFFFFF;
        let file_crc = read_u32(file)?;
        if file_crc != crc {
            return Err(PngError::InvalidChunkCRC(file_crc, crc));
        }

        let bytes_per_pixel = metadata::bytes_per_pixel(colour_type) as usize;
        let header = Self {
            width,
            height,
            bytes_per_pixel,
        };

        Ok(header)
    }
}

#[repr(C)]
pub struct HeliumPngData {
    pub width: u32,
    pub height: u32,
    pub num_channels: u32,
    pub data: *mut c_void,
}

struct HuffmanCode {
    codes: [u16; 16],
    map: [Vec<u16>; 16],
}

impl HuffmanCode {
    fn decode(&self, bit_buffer: &mut BitBuffer) -> Result<u16, ZlibError> {
        let mut code = 0;
        for bit in 0..15 {
            code = (code << 1) | bit_buffer.bits(1)?;
            let start = self.codes.get(bit + 1).ok_or(ZlibError::InvalidBitLength)?;
            let bit_codes = self.map.get(bit + 1).ok_or(ZlibError::InvalidBitLength)?;
            if let Some(&value) = bit_codes.get((code - start) as usize) {
                return Ok(value);
            }
        }

        Err(ZlibError::InvalidHuffmanCode)
    }

    fn from_dynamic(hclen: usize, bit_buffer: &mut BitBuffer) -> Result<Self, ZlibError> {
        let code_indices: [_; 19] = [
            16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
        ];
        let mut cc_bit_lengths = [0; 19];

        for i in 0..hclen {
            // None case can only happen due to PNG corruption as we've made the `code_indices` array
            // the size of the maximum possible elements allowable by the standard.
            let index = code_indices.get(i).ok_or(ZlibError::InvalidHCLEN)?;
            // None case cannot happen, by construction. We avoid panicking here.
            if let Some(value) = cc_bit_lengths.get_mut(*index as usize) {
                *value = bit_buffer.bits(3)? as u8;
            }
        }

        Ok(Self::new(&cc_bit_lengths))
    }

    /// Creates a new set of Huffman codes from an alphabet.
    ///
    /// Takes a slice of unsigned 8-bit integers where each integer's index is its alphabet value
    /// and where each integer's value is the bit length of that alphabet value. Thus, this
    /// function implicitly assumes that `code_bit_lengths` is in lexicographic order of the
    /// DEFLATE alphabet.
    fn new(code_bit_lengths: &[u8]) -> Self {
        let mut counts = [0; 16];
        let mut map: [Vec<u16>; 16] = Default::default();
        for (i, &bit_length) in code_bit_lengths.iter().enumerate() {
            let bit_length = bit_length as usize;
            if let Some(c) = counts.get_mut(bit_length) {
                *c += 1;
            }
            if let Some(m) = map.get_mut(bit_length) {
                m.push(i as u16);
            }
        }

        // It is not possible for there to be a code with zero bits. Therefore, we set the count
        // of all codes with bit length zero to zero.
        counts[0] = 0;
        map[0].clear();

        // Compiler's static analysis should eliminate bounds checks and panic paths for this case
        let mut codes = [0; 16];
        let mut start = 0;
        for bit in 1..16 {
            start = (start + counts[bit - 1]) << 1;
            codes[bit] = start;
        }

        Self { codes, map }
    }
}

pub struct Image {
    pub width: u32,
    pub height: u32,
    pub num_channels: u32,
    pub data: Vec<u8>,
}

#[repr(C)]
pub struct HeliumImageMetadata {
    width: u32,
    height: u32,
    bits_per_pixel: u32,
}

#[derive(Debug)]
enum MetadataError {
    FileOpenError = 1,
    FileReadError = 2,
    InvalidMagicNumber = 3,
    InvalidHeader = 4,
}

impl Display for MetadataError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "")
    }
}

impl Error for MetadataError {}

impl From<std::io::Error> for MetadataError {
    fn from(e: std::io::Error) -> Self {
        use std::io::ErrorKind;

        match e.kind() {
            ErrorKind::NotFound | ErrorKind::PermissionDenied => Self::FileOpenError,
            _ => Self::FileReadError,
        }
    }
}

#[derive(Debug)]
enum PngError {
    InvalidChunkCRC(u32, u32),
    InvalidHeader,
    InvalidMagicNumber,
    NonZeroCompressionMethod,
    NonZeroFilterMethod,
    PartialOrOverReconstruction,
    FileReadError,
    UnknownFilterType,
    UnsupportedBitDepth(u8),
    UnsupportedInterlaceMethod,
}

impl Display for PngError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "")
    }
}

impl Error for PngError {}

impl From<std::io::Error> for PngError {
    fn from(_: std::io::Error) -> Self {
        PngError::FileReadError
    }
}

struct Timer {
    now: std::time::Instant,
}

impl Timer {
    fn checkpoint(&mut self) -> std::time::Duration {
        let now = std::time::Instant::now();
        let duration = now - self.now;
        self.now = now;

        duration
    }

    fn new() -> Self {
        Self {
            now: std::time::Instant::now(),
        }
    }
}

#[derive(Debug)]
enum ZlibError {
    InvalidBitLength,
    InvalidCodeLength,
    InvalidCompressionMethod,
    InvalidDistance,
    InvalidFCHECK,
    InvalidHCLEN,
    InvalidHuffmanCode,
    InvalidLiteralLength,
    InvalidRawBlock,
    InvalidWindowLength,
    PartialStreamInflation,
    PresetDictionaryPresent,
    StreamOverflow,
    UnknownBlockCompression,
}

impl Display for ZlibError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "")
    }
}

impl Error for ZlibError {}

fn decode_block<'b, 's>(
    bit_buffer: &'b mut BitBuffer<'s>,
    decoded_stream: &'b mut Vec<u8>,
    distance: &'b HuffmanCode,
    literal_length: &'b HuffmanCode,
) -> Result<(), ZlibError> {
    loop {
        match literal_length.decode(bit_buffer)? {
            x @ 0..=255 => decoded_stream.push(x as u8),
            256 => break Ok(()),
            x @ 257..=285 => {
                let length_index = x as usize - 257;
                let extra_bits = LENGTH_EXTRA_BITS[length_index];
                let extra_length = bit_buffer.bits(extra_bits)?;
                let length = LENGTHS[length_index] + extra_length;

                // Get distance code
                let distance_index = match distance.decode(bit_buffer) {
                    Ok(x @ 0..=29) => x as usize,
                    _ => break Err(ZlibError::InvalidDistance),
                };
                let extra_bits = DISTANCE_EXTRA_BITS[distance_index];
                let extra_distance = bit_buffer.bits(extra_bits)?;
                let distance = DISTANCES[distance_index] + extra_distance;

                let current_length = decoded_stream.len();
                for i in 0..length {
                    let index = current_length - distance as usize + i as usize;
                    let value = decoded_stream[index];
                    decoded_stream.push(value);
                }
            }
            _ => break Err(ZlibError::InvalidLiteralLength),
        }
    }
}

/// Decodes a dynamic Huffman code block.
fn dynamic<'b, 's>(
    bit_buffer: &'b mut BitBuffer<'s>,
    decoded_stream: &'b mut Vec<u8>,
) -> Result<(), ZlibError> {
    let header = bit_buffer.bits(14)? as usize;
    let hlit = (header & 0x1F) + 257;
    let hdist = ((header >> 5) & 0x1F) + 1;
    let hclen = (header >> 10) + 4;

    let cc = HuffmanCode::from_dynamic(hclen, bit_buffer)?;
    let mut code_lengths = vec![0; hlit + hdist];
    let mut num_decoded = 0;
    let mut last_code = 0;

    while num_decoded < code_lengths.capacity() {
        let (repeat, code_to_repeat) = match cc.decode(bit_buffer)? {
            x @ 0..=15 => (1, x as u8),
            16 => (3 + bit_buffer.bits(2)?, last_code),
            17 => (3 + bit_buffer.bits(3)?, 0),
            18 => (11 + bit_buffer.bits(7)?, 0),
            _ => return Err(ZlibError::InvalidCodeLength),
        };

        for _ in 0..repeat {
            code_lengths[num_decoded] = code_to_repeat;
            num_decoded += 1;
        }
        last_code = code_to_repeat;
    }

    let mut ll_bit_lengths = [0; 287];
    ll_bit_lengths[..hlit].copy_from_slice(&code_lengths[..hlit]);
    let mut distance_bit_lengths = [0; 32];
    distance_bit_lengths[..hdist].copy_from_slice(&code_lengths[hlit..]);

    let literal_length = HuffmanCode::new(&ll_bit_lengths);
    let distance = HuffmanCode::new(&distance_bit_lengths);
    decode_block(bit_buffer, decoded_stream, &distance, &literal_length)
}

/// Direct port of the CRC table generation algorithm given in the PNG specification:
/// https://www.w3.org/TR/PNG/#D-CRCAppendix
fn generate_crc_table() -> [u32; NUM_8BIT_CRCS] {
    let mut table = [0; NUM_8BIT_CRCS];

    for (mut i, c) in table.iter_mut().enumerate() {
        for _ in 0..8 {
            i = if (i & 1) == 1 {
                0xEDB88320 ^ (i >> 1)
            } else {
                i >> 1
            }
        }

        *c = i as u32;
    }

    table
}

pub fn helium(file_name: &str) -> Result<Image, Box<dyn Error>> {
    let mut file = File::open(file_name)?;
    let mut buffer = [0; BUFFER_SIZE];

    // Generate a table of 32-bit CRC's of all possible 8-bit values.
    let crc_table = generate_crc_table();

    let mut timer = Timer::new();

    // First 8-bytes of a PNG file are its identifying magic number
    if !metadata::contains_magic_number(&mut file) {
        return Err(Box::new(PngError::InvalidMagicNumber));
    }
    let header = Header::parse(&crc_table, &mut file)?;

    // Collect zlib stream
    let mut zlib_stream = Vec::new();
    let mut chunk_type = IHDR;
    while chunk_type != IEND {
        let length = read_u32(&mut file)? as usize;
        file.read_exact(&mut buffer[..4])?;
        chunk_type.copy_from_slice(&buffer[..4]);

        let mut crc = update_crc(0xFFFF_FFFF, &buffer[..4], &crc_table);

        let mut bytes_read = 0;
        while bytes_read < length {
            let bytes_to_read = usize::min(length - bytes_read, BUFFER_SIZE);
            let buffer = &mut buffer[..bytes_to_read];
            file.read_exact(buffer)?;
            crc = update_crc(crc, buffer, &crc_table);
            bytes_read += bytes_to_read;

            if chunk_type == IDAT {
                zlib_stream.extend_from_slice(buffer);
            }
        }

        // Need to invert the bits in the calculated CRC at the end so we XOR with 0xFFFFFFFF.
        crc ^= 0xFFFF_FFFF;
        let file_crc = read_u32(&mut file)?;
        if file_crc != crc {
            return Err(Box::new(PngError::InvalidChunkCRC(file_crc, crc)));
        }
    }
    println!(
        "Parsing and collection finished in {:?}",
        timer.checkpoint()
    );

    let inflated_stream = inflate(&zlib_stream, &header)?;
    println!("Inflation finished in {:?}", timer.checkpoint());
    let image_data = reconstruct(&inflated_stream, &header)?;
    println!("Reconstruction finished in {:?}", timer.checkpoint());

    Ok(Image {
        width: header.width as u32,
        height: header.height as u32,
        num_channels: header.bytes_per_pixel as u32,
        data: image_data,
    })
}

#[no_mangle]
pub extern "system" fn helium_get_metadata(
    #[cfg(windows)] file_name: *const u16,
    #[cfg(not(windows))] file_name: *const u8,
    metadata: *mut HeliumImageMetadata,
) -> u32 {
    if file_name.is_null() || metadata.is_null() {
        return 1;
    }

    let name = {
        let length = (0..)
            .take_while(|&i| unsafe { *file_name.offset(i) } != 0)
            .count();
        let slice = ManuallyDrop::new(unsafe { slice::from_raw_parts(file_name, length) });
        #[cfg(windows)]
        {
            use std::os::windows::ffi::OsStringExt;
            std::ffi::OsString::from_wide(*slice)
        }
        #[cfg(not(windows))]
        {
            use std::os::unix::ffi::OsStrExt;
            std::ffi::OsStr::from_bytes(*slice)
        }
    };

    match name.to_str() {
        Some(file_name) => match metadata::get(file_name) {
            Ok(m) => unsafe {
                *metadata = m;
            },
            // Add 1 since we already use 1 and the enum discriminant starts at 1.
            Err(e) => return (e as u32) + 1,
        },
        None => return 3,
    }

    0
}

#[no_mangle]
pub extern "system" fn helium_decode_png(
    #[cfg(windows)] file_name: *const u16,
    #[cfg(not(windows))] file_name: *const u8,
    png_data: *mut HeliumPngData,
) -> u32 {
    if file_name.is_null() {
        return 1;
    } else if png_data.is_null() {
        return 2;
    }

    let name = {
        let length = (0..)
            .take_while(|&i| unsafe { *file_name.offset(i) } != 0)
            .count();
        let slice = ManuallyDrop::new(unsafe { slice::from_raw_parts(file_name, length) });
        #[cfg(windows)]
        {
            use std::os::windows::ffi::OsStringExt;
            std::ffi::OsString::from_wide(*slice)
        }
        #[cfg(not(windows))]
        {
            use std::os::unix::ffi::OsStrExt;
            std::ffi::OsStr::from_bytes(*slice)
        }
    };

    if let Some(n) = name.to_str() {
        if let Ok(i) = helium(n) {
            // Wrap data in a `ManuallyDrop` in order to avoid calling the destructor.
            // `std::mem::forget` can still lead to UB due to compiler assumptions!
            let mut d = ManuallyDrop::new(i.data);
            let p = HeliumPngData {
                width: i.width,
                height: i.height,
                num_channels: i.num_channels,
                data: d.as_mut_ptr() as *mut c_void,
            };

            unsafe {
                *png_data = p;
            }
        } else {
            // Failed to decode the PNG file
            return 4;
        }
    } else {
        // Failed to create a valid UTF-8 encoded name from the given file name
        return 3;
    }

    0
}

/// Attempts to decode (inflate) a stream encoded by the DEFLATE codec.
fn inflate(zlib_stream: &[u8], header: &Header) -> Result<Vec<u8>, ZlibError> {
    // We need to start by validating the first two bytes of the ZLIB stream.
    // First comes the compression method and flags (CMF) byte. Bits 0-3 (4-bits) store the
    // compression method (CM) and bits 4-7 store the compression info (CINFO).
    let cmf = zlib_stream[0];

    // A CM value of 8 indicate the "DEFLATE" compression method with a window size up to 32Kb. This
    // is the only method defined by the specification, hence, CM can only be 8.
    let cm = cmf & 0xF;
    if cm != 8 {
        return Err(ZlibError::InvalidCompressionMethod);
    }

    // CINFO is the base-2 logarithm of of the LZ77-compression window size, minus eight used to
    // encode this DEFLATE stream. Values of CINFO above 7 are not allowed by the specification as
    // 2^(7 + 8) = 32767. Storing this value is not neccessary.
    let cinfo = cmf >> 4;
    if cinfo > 7 {
        return Err(ZlibError::InvalidWindowLength);
    }

    // Second comes the flags (FLG) byte. Bits 0-4 (5-bits) specify the check bits for CMF and FLG
    // (FCHECK), bit 5 specifies whether there is a preset dictionary present (FDICT), and bits 6-7
    // indicate the compression level (FLEVEL) - the FLEVEL value can be safely ignored, hence, we do
    // not bother checking it.
    let flg = zlib_stream[1];

    // FCHECK must be a value such that when CMF and FLG are viewed as a 16-bit unsigned integer
    // stored in MSB order, FCHECK is a multiple of 31. Hence, we do not directly check the FCHECK
    // value.
    let integer = ((cmf as u16) << 8) | (flg as u16);
    if (integer % 31) != 0 {
        return Err(ZlibError::InvalidFCHECK);
    }

    // Currently we do not support decompression in the presence of a preset dictionary as this is not
    // required by the PNG specification.
    let fdict = (flg & 0x20) >> 5;
    if fdict == 1 {
        return Err(ZlibError::PresetDictionaryPresent);
    }

    let mut bit_buffer = BitBuffer {
        bits: 0,
        bit_count: 0,
        index: 0,
        length: zlib_stream.len() - 2,
        stream: &zlib_stream[2..],
    };

    // Total allocation = Total number of pixels + additional filter bytes for each scanline (the + 1)
    let num_bytes = (header.bytes_per_pixel * header.width + 1) * header.height;
    let mut decoded_stream = Vec::with_capacity(num_bytes);

    let mut last_block = false;
    while !last_block {
        // The first three bits are a part of the block header. Bit
        // Bit 0 indicates whether the block is the last or not
        // Bits 1-2 indicate the type of the block:
        // - 0 => Raw data block
        // - 1 => Block stored with Fixed Huffman Code compression
        // - 2 => Block stored with Dynamic Huffman Code compression
        last_block = bit_buffer.bits(1)? == 1;
        match bit_buffer.bits(2)? {
            0 => {
                // The fixed block starts on the next byte boundary. We have just used 3 bits to
                // get here, now we throw away the rest of the 5 bits in this byte.
                bit_buffer.throw_bits(5);

                // The raw data block is defined by two bytes:
                // 1. Length of the block
                // 2. One's complement of the length. This is used to ensure that we do not
                // misinterpret the start of a raw block.
                // 3. Length bytes of raw pixel data
                let length = bit_buffer.bits(16)?;
                let ones_complement = bit_buffer.bits(16)?;
                if (length ^ 0xFFFF) != ones_complement {
                    return Err(ZlibError::InvalidRawBlock);
                }

                for _ in 0..length {
                    let value = bit_buffer.bits(8)? as u8;
                    decoded_stream.push(value);
                }
            }
            1 => todo!(),
            2 => dynamic(&mut bit_buffer, &mut decoded_stream)?,
            _ => return Err(ZlibError::UnknownBlockCompression),
        }
    }

    if decoded_stream.len() != num_bytes {
        Err(ZlibError::PartialStreamInflation)
    } else {
        Ok(decoded_stream)
    }
}

/// A port of the paeth filter described in the PNG specification:
/// https://www.w3.org/TR/PNG/#9Filter-type-4-Paeth
///
/// It attempts to compute a linear function of the neighbouring bytes of a given byte. Given
/// a byte `x`, let `a` be the byte immediately before x, let `b` be the byte in x's position in
/// the previous scanline, and let `c` be the byte in a's position in the previous scanline. This
/// function tries to find in which of the three direction's (vertical, horizontal, and diagonal)
/// the gradient is the smallest.
fn paeth(a: u8, b: u8, c: u8) -> u8 {
    let a16 = a as i16;
    let b16 = b as i16;
    let c16 = c as i16;

    let p = a16 + b16 - c16;
    let pa = i16::abs(p - a16);
    let pb = i16::abs(p - b16);
    let pc = i16::abs(p - c16);

    if (pa <= pb) && (pa <= pc) {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
}

fn read_u32(f: &mut File) -> std::io::Result<u32> {
    let mut b = [0; 4];
    f.read_exact(&mut b[..])?;

    Ok(u32::from_be_bytes(b))
}

fn read_u8(f: &mut File) -> std::io::Result<u8> {
    let mut b = [0; 1];
    f.read_exact(&mut b[..])?;

    Ok(b[0])
}

fn reconstruct(inflated_stream: &[u8], header: &Header) -> Result<Vec<u8>, PngError> {
    let bppo = header.bytes_per_pixel + 1;
    let scanline_width = header.bytes_per_pixel * header.width;
    let mut unfiltered_stream = Vec::with_capacity(scanline_width * header.height);
    let filter_byte_index = scanline_width + 1;

    // We start to reconstruct the PNG image from the inflated ZLIB (DEFLATE) stream. There are five
    // filter methods present in the PNG specification:
    // 0 (None) -> ,
    // 1 (Sub) -> ,
    // 2 (Up) -> ,
    // 3 (Average) -> ,
    // 4 (Paeth) -> ,
    match inflated_stream[0] {
        0 | 2 => unfiltered_stream.extend_from_slice(&inflated_stream[1..filter_byte_index]),
        1 => {
            unfiltered_stream.extend_from_slice(&inflated_stream[1..bppo]);
            for byte in bppo..filter_byte_index {
                let x = inflated_stream[byte];
                let a = unfiltered_stream[byte - bppo];
                let (value, _) = x.overflowing_add(a);
                unfiltered_stream.push(value);
            }
        }
        3 => {
            // First pixel: We don't have pixel to the left and we don't have a pixel to the top.
            unfiltered_stream.extend_from_slice(&inflated_stream[1..bppo]);
            // Rest: Only have a pixel to the left, not above.
            for byte in bppo..scanline_width {
                let (value, _) =
                    inflated_stream[byte].overflowing_add(unfiltered_stream[byte - bppo]);
                unfiltered_stream.push(value >> 1);
            }
        }
        4 => todo!("Paeth"),
        _ => return Err(PngError::UnknownFilterType),
    };

    for scanline in 1..header.height {
        let filter_byte = scanline * filter_byte_index;
        let first_byte = filter_byte + 1;
        let unfiltered_cs = scanline * scanline_width;
        let unfiltered_ps = (scanline - 1) * scanline_width;

        match inflated_stream[filter_byte] {
            0 => unfiltered_stream.extend_from_slice(
                &inflated_stream[(filter_byte + 1)..(filter_byte + filter_byte_index)],
            ),
            1 => {
                unfiltered_stream.extend_from_slice(
                    &inflated_stream[first_byte..(first_byte + header.bytes_per_pixel)],
                );
                for byte in bppo..filter_byte_index {
                    let x = inflated_stream[filter_byte + byte];
                    let a = unfiltered_stream[unfiltered_cs + byte - bppo];
                    let (value, _) = x.overflowing_add(a);
                    unfiltered_stream.push(value);
                }
            }
            2 => {
                for byte in 0..scanline_width {
                    let x = inflated_stream[first_byte + byte];
                    let b = unfiltered_stream[unfiltered_ps + byte];
                    let (value, _) = x.overflowing_add(b);
                    unfiltered_stream.push(value);
                }
            }
            3 => {
                for byte in 0..header.bytes_per_pixel {
                    let x = inflated_stream[first_byte + byte];
                    let b = unfiltered_stream[unfiltered_ps + byte];
                    let (value, _) = x.overflowing_add(b >> 1);
                    unfiltered_stream.push(value);
                }

                for byte in bppo..filter_byte_index {
                    let x = inflated_stream[filter_byte + byte];
                    let b = unfiltered_stream[unfiltered_ps + byte - 1] as u16;
                    let a = unfiltered_stream[unfiltered_cs + byte - bppo] as u16;

                    let (value, _) = x.overflowing_add(((a + b) >> 1) as u8);
                    unfiltered_stream.push(value);
                }
            }
            4 => {
                for byte in 0..header.bytes_per_pixel {
                    let x = inflated_stream[first_byte + byte];
                    let b = unfiltered_stream[unfiltered_ps + byte];
                    let (value, _) = x.overflowing_add(paeth(0, b, 0));
                    unfiltered_stream.push(value);
                }

                for byte in bppo..filter_byte_index {
                    let x = inflated_stream[filter_byte + byte];
                    let c = unfiltered_stream[unfiltered_ps + byte - bppo];
                    let b = unfiltered_stream[unfiltered_ps + byte - 1];
                    let a = unfiltered_stream[unfiltered_cs + byte - bppo];

                    let (value, _) = x.overflowing_add(paeth(a, b, c));
                    unfiltered_stream.push(value);
                }
            }
            _ => return Err(PngError::UnknownFilterType),
        }
    }

    if unfiltered_stream.len() != (scanline_width * header.height) {
        Err(PngError::PartialOrOverReconstruction)
    } else {
        Ok(unfiltered_stream)
    }
}

/// Port of the CRC algorithm descibed in the PNG specification -
/// https://www.w3.org/TR/PNG/#D-CRCAppendix
fn update_crc(mut crc: u32, buffer: &[u8], crc_table: &[u32; 256]) -> u32 {
    for i in buffer {
        crc = crc_table[(crc ^ *i as u32) as usize & 0xFF] ^ (crc >> 8);
    }

    crc
}
