#ifndef HELIUM_H
#define HELIUM_H

#if defined _WIN32 || defined WIN32
    int helium_c(unsigned short* file_name, int* width, int* height, int* num_channels, unsigned char** data);
#elif defined __unix__ || defined __MACH__
    int helium_c(char* file_name, int* width, int* height, int* num_channels, unsigned char** data);
#endif // !(defined _WIN32 || defined WIN32) && !(defined __unix__ || defined __MACH__)

#endif // HELIUM_H
