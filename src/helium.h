#ifndef HELIUM_H
#define HELIUM_H

long helium_c(
#if defined _WIN32 || defined WIN32
    short *file_name, // Windows uses UTF-16 for strings. We support that natively.
#else // !(defined _WIN32 || defined WIN32)
    char *file_name, // All other cases use UTF-8
#endif // defined _WIN32 || defined WIN32
    unsigned long *width,
    unsigned long *height,
    long *num_channels,
    unsigned char **data
);

#endif // HELIUM_H
