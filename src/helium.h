#ifndef HELIUM_H
#define HELIUM_H

typedef struct HeliumImageMetadata {
    unsigned int width;
    unsigned int height;
    unsigned int bits_per_pixel;
} HeliumImageMetadata;

typedef struct HeliumPngData {
    unsigned int width;
    unsigned int height;
    unsigned int num_channels;
    void*        data;
} HeliumPngData;

// Attempts to read the PNG file `file_name` and retrieve its width, height, and bits per pixel.
// The metadata is filled into the `ImageMetadata` struct once retrieved. Once done, the image
// data can simply be de-allocated using `free` on unix systems or `HeapFree` on Windows.
// Return codes:
// 0 - on success
// 1 - if `filename` is NULL,
// 2 - if `metadata` is NULL,
// 3 - if `filename` is not valid UTF-8
// 4 - if an error occurred while opening `filename`
// 5 - if an error occurred while reading `filename`
// 6 - if `filename` is not a valid PNG file (invalid PNG magic number)
// 7 - if the PNG header in `filename` is invalid
unsigned int helium_decode_png(
#if defined(_WIN32) || defined(_WIN64)
    const unsigned short* file_name,
#else
    const char* file_name,
#endif
    HeliumPngData* png_data
);

unsigned int helium_get_metadata(
#if defined(_WIN32) || defined(_WIN64)
    const unsigned short* filename,
#else
    const char* filename,
#endif
    HeliumImageMetadata* metadata
);

#endif // HELIUM_H
