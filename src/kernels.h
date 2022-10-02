#ifndef _KERNELS_H_
#define _KERNELS_H_

#include <cuComplex.h>

#include "common.h"

void FftShiftGpu(cuComplex *data, int num_samples, int num_arrays,
                 cudaStream_t stream);

size_t GetBackprojWorkBufSizeBytes(SarGpuKernel kernel, int num_range_bins);

void SarBpGpu(cuComplex *image, int image_width, int image_height,
              const cuComplex *range_profiles, uint8_t *bp_workbuf,
              int num_range_bins, int num_pulses, const float3 *ant_pos,
              double freq_min, double dr, double dx, double dy, double z0,
              SarGpuKernel selected_kernel, cudaStream_t stream);

size_t GetMaxMagnitudeWorkBufSizeBytes(int image_height);

void ComputeMagnitudeImage(uint *dest_image, float *max_magnitude_workbuf,
                           const cuComplex *src_image, int image_width,
                           int image_height, float min_normalized_db,
                           cudaStream_t stream);

#endif /* _KERNELS_H_ */
