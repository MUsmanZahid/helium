#ifndef HELIUM_H
#define HELIUM_H

int helium_c(
    unsigned
    #if defined _WIN32 || defined WIN32
    short
    #else // !(defined _WIN32 || defined WIN32)
    char
    #endif // defined _WIN32 || defined WIN32
    *file_name, int *width, int *height, int *num_channels, unsigned char **data
);

#endif // HELIUM_H
