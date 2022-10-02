#include "helpers.h"

#include <cstdarg>
#include <cstdio>
#include <cstring>

#define MAX_LOGGED_BYTES (1024)

void __log(const char *fmt, ...) {
    va_list argp;
    va_start(argp, fmt);
    char msg[MAX_LOGGED_BYTES];
    vsnprintf(msg, sizeof(msg), fmt, argp);
    va_end(argp);
    printf("%s\n", msg);
}

void __logErr(const char *file, int line, const char *fmt, ...) {
    va_list argp;
    va_start(argp, fmt);
    char msg[MAX_LOGGED_BYTES];
    snprintf(msg, sizeof(msg), "%s:%d: ", file, line);
    vsnprintf(&msg[strlen(msg)], sizeof(msg) - strlen(msg), fmt, argp);
    va_end(argp);
    printf("%s\n", msg);
}

void __cudaChecked(cudaError err, const char *file, int line) {
    if (err == cudaSuccess) {
        return;
    }
    LOG("%s:%d : CUDA Runtime API error: %s", file, line,
        cudaGetErrorString(err));
    exit(EXIT_FAILURE);
}

void __cufftChecked(cufftResult err, const char *file, int line) {
    if (err == CUFFT_SUCCESS) {
        return;
    }
    LOG("%s:%d : CUFFT API error: %d", file, line, err);
    exit(EXIT_FAILURE);
}
