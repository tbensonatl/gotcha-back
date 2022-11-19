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

    SarBpGpu(int num_range_bins);
    ~SarBpGpu();

    // Backproject num_pulses x m_num_range_bins pulses from dev_range_profiles
    // into dev_image cooresponding to antenna positions dev_ant_pos. The output
    // image has pixel dimensions image_width x image_height and physical
    // dimensions image_fov_m x image_fov_m meters. The image is not zeroed out
    // prior to backprojection so that pulses can be accumulated with multiple
    // calls to Backproject. stream is not synchronized in this method, so the
    // backprojection will likely not be complete yet when this method returns.
    void Backproject(cuComplex *dev_image, const cuComplex *dev_range_profiles,
                     const float3 *dev_ant_pos, int image_width,
                     int image_height, int num_pulses, float image_fov_m,
                     float min_freq, float del_freq, SarGpuKernel kernel,
                     cudaStream_t stream);

  private:
    int m_num_range_bins;

    // Device buffers
    uint8_t *m_dev_workbuf{nullptr};
};

#endif // _SAR_BP_GPU_H_