#include "sar_bp_gpu.h"
#include "helpers.h"
#include "kernels.h"

#include <chrono>

SarBpGpu::SarBpGpu(int num_range_bins) : m_num_range_bins(num_range_bins) {
    const size_t bp_workbuf_size_bytes =
        GetMaxBackprojWorkBufSizeBytes(m_num_range_bins);
    cudaChecked(cudaMalloc((void **)&m_dev_workbuf, bp_workbuf_size_bytes));
}

SarBpGpu::~SarBpGpu() { FREE_AND_NULL_CUDA_DEV_ALLOC(m_dev_workbuf); }

void SarBpGpu::Backproject(cuComplex *dev_image,
                           const cuComplex *dev_range_profiles,
                           const float3 *dev_ant_pos, int image_width,
                           int image_height, int num_pulses, float image_fov_m,
                           float min_freq, float del_freq, SarGpuKernel kernel,
                           cudaStream_t stream) {
    const double max_wr = SPEED_OF_LIGHT_M_PER_SEC / (2.0 * del_freq);
    const double dx = static_cast<double>(image_fov_m) / image_width;
    const double dy = static_cast<double>(image_fov_m) / image_height;
    const double dr = max_wr / m_num_range_bins;

    SarBpGpuWrapper(dev_image, image_width, image_height, dev_range_profiles,
                    m_dev_workbuf, m_num_range_bins, num_pulses, dev_ant_pos,
                    min_freq, dr, dx, dy, 0.0f, kernel, stream);
    cudaChecked(cudaGetLastError());
}
