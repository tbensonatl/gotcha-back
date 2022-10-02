#ifndef _HELPERS_H_
#define _HELPERS_H_

#include <cuda_runtime_api.h>
#include <cufft.h>

#define LOG(fmt, ...) __log(fmt, ##__VA_ARGS__)
#define LOG_ERR(fmt, ...) __logErr(__FILE__, __LINE__, fmt, ##__VA_ARGS__)

#define cudaChecked(err) __cudaChecked(err, __FILE__, __LINE__)
#define cufftChecked(err) __cufftChecked(err, __FILE__, __LINE__)

void __log(const char *fmt, ...);
void __logErr(const char *file, int line, const char *fmt, ...);

void __cudaChecked(cudaError err, const char *file, int line);
void __cufftChecked(cufftResult err, const char *file, int line);

#endif /* _HELPERS_H_ */