#ifndef _KERNELS_H_
#define _KERNELS_H_

#include <cuComplex.h>

#include "common.h"

void FftShiftGpu(cuComplex *data, int num_samples, int num_arrays,
                 cudaStream_t stream);

// max_num_pulses is the maximum number of pulses for which SarBpGpuWrapper
// will be invoked. Pulse counts larger than max_num_pulses can be accommodated
// by multiple calls to SarBpGpuWrapper, each with no more than max_num_pulses
// pulses
size_t GetMaxBackprojWorkBufSizeBytes(int num_range_bins, int max_num_pulses);

void SarBpGpuWrapper(cuComplex *image, int image_width, int image_height,
                     cudaTextureObject_t tex_obj,
                     const cuComplex *range_profiles, uint8_t *bp_workbuf,
                     int num_range_bins, int num_pulses, const float3 *ant_pos,
                     double freq_min, double dr, double dx, double dy,
                     double z0, SarGpuKernel selected_kernel,
                     cudaStream_t stream);

size_t GetMaxMagnitudeWorkBufSizeBytes(int image_height);

void ComputeMagnitudeImage(uint32_t *dest_image, float *max_magnitude_workbuf,
                           const cuComplex *src_image, int image_width,
                           int image_height, float min_normalized_db,
                           cudaStream_t stream);

void ResampleMagnitudeImage(uint32_t *resampled, const uint32_t *src_image,
                            int resample_width, int resample_height,
                            int src_width, int src_height, cudaStream_t stream);

#endif /* _KERNELS_H_ */
