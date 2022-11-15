#ifndef _SAR_BP_GPU_H_
#define _SAR_BP_GPU_H_

#include <cstdint>

#include <cuComplex.h>
#include <cuda_runtime_api.h>

#include "common.h"

class SarBpGpu {
  public:
    SarBpGpu() = delete;
    SarBpGpu(const SarBpGpu &) = delete;
    SarBpGpu &operator=(const SarBpGpu &) = delete;

    SarBpGpu(int num_range_bins, int max_image_num_pixels, cudaStream_t stream);
    ~SarBpGpu();

    // Backproject does not synchronize the stream
    const cuComplex *Backproject(const cuComplex *dev_range_profiles,
                                 const float3 *dev_ant_pos, int image_width,
                                 int image_height, int num_pulses,
                                 float image_fov_m, float min_freq,
                                 float del_freq, SarGpuKernel kernel);

  private:
    int m_num_range_bins;
    int m_max_image_num_pixels;

    // Device buffers
    uint8_t *m_dev_workbuf{nullptr};
    cuComplex *m_dev_image{nullptr};

    // m_stream is not owned by this class
    cudaStream_t m_stream;
};

#endif // _SAR_BP_GPU_H_