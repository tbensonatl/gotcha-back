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

    // max_num_pulses is the maximum number of pulses for which Backproject()
    // will be called; it is used to size a work buffer. A caller can integrate
    // more than max_num_pulses with multiple calls to Backproject(), each
    // having at most max_num_pulses pulses.
    SarBpGpu(int num_range_bins, int max_num_pulses);
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
    bool KernelUsesTextureMemory(SarGpuKernel kernel);

    int m_num_range_bins;
    int m_max_num_pulses;

    // Device buffers
    uint8_t *m_dev_workbuf{nullptr};
    cudaArray *m_dev_tex_array{nullptr};
    cudaTextureObject_t m_tex_obj{0};
};

#endif // _SAR_BP_GPU_H_