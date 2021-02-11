# Helium
Helium is a PNG loading library that aims to be fast and simple to read. It currently supports
decoding 8- and 16-bit greyscale and RGB PNG images. This library requires an allocator to be
present and it currenly requires support of the the rust standard library on the platform it is
being deployed on.

## Why?
This library was built in my spare time, mainly to learn the technology and techniques behind how
PNG images are stored and displayed, as well as to apply performance engineering concpets I've
learned in lectures and talks.

## Usage
Along with the Rust entry-point, a C/C++ entry-point is also provided through FFI. Both interface
take a file name, open the file, and attempt to decode the PNG image. Allocations are all done
within the library.

### Rust
```Rust
fn helium(file_name: &str) -> Result<Image, ZlibError>
```

### C/C++
A header file called `helium.h` is provided in the *src* directory. It defines one function with
two signatures depending on the platform. For Windows, it supports its native UTF-16 interface for
file names, and UTF-8 for all other platforms.
```C
long helium_c(
#if defined _WIN32 || defined WIN32
    short *file_name, // UTF-16
#else // !(defined _WIN32 || defined WIN32)
    char *file_name, // UTF-8
#endif // defined _WIN32 || defined WIN32
    unsigned long *width,
    unsigned long *height,
    long *num_channels,
    unsigned char **data
);
```

## Features
- A simple interface
- Easy to read
- Less than 1000 SLOC
- Dependency free, only Rust standard library
- Fast decoding of PNG files with:
  - 8- and 16-bit colour depth
  - Greyscale and greyscale with alpha colour channels
  - RGB and RGBA colour channels

## Roadmap
- [ ] Move to a `#[no_std]` implementation while using the `alloc` crate.
- [ ] Implement Fixed Huffman block decoding
- [ ] Support a separate interface to interact with _just_ the DEFLATE decompressor
- [ ] Add an interface which allows call-site determination of allocation type (stack or heap).
      Useful for small images.
  - [ ] Implement a ring buffer decoding strategy to avoid unnecessary allocation during decoding.
