#ifndef HELIUM_H
#define HELIUM_H

typedef struct ImageMetadata {
    unsigned int width;
    unsigned int height;
    unsigned int bits_per_pixel;
} ImageMetadata;

// Attempts to read the PNG file `filename` and retrieve its width, height, and bits per pixel.
// The metadata is filled into the `ImageMetadata` struct once retrieved.
// Return codes:
// 0 - on success
// 1 - if `filename` is NULL,
// 2 - if `metadata` is NULL,
// 3 - if `filename` is not valid UTF-8
// 4 - if an error occurred while opening `filename`
// 5 - if an error occurred while reading `filename`
// 6 - if `filename` is not a valid PNG file (invalid PNG magic number)
// 7 - if the PNG header in `filename` is invalid
unsigned int helium_get_metadata(
#if defined(_WIN32) || defined(_WIN64)
    const unsigned short* filename,
#else
    const char* filename,
#endif
    ImageMetadata* metadata
);

long helium_c(
#if defined(_WIN32) || defined(_WIN64)
    unsigned short* filename, // Windows uses UTF-16 for strings. We support that natively.
#else
    char* filename, // All other cases use UTF-8
#endif
    unsigned long* width,
    unsigned long* height,
    long* num_channels,
    void** data
);

#endif // HELIUM_H
