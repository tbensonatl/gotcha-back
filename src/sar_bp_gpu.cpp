#include "sar_bp_gpu.h"
#include "helpers.h"
#include "kernels.h"

#include <chrono>
#include <cstring>

SarBpGpu::SarBpGpu(int num_range_bins, int max_num_pulses)
    : m_num_range_bins(num_range_bins), m_max_num_pulses(max_num_pulses) {
    const size_t bp_workbuf_size_bytes =
        GetMaxBackprojWorkBufSizeBytes(m_num_range_bins, max_num_pulses);
    cudaChecked(cudaMalloc((void **)&m_dev_workbuf, bp_workbuf_size_bytes));

    cudaChannelFormatDesc channel_desc =
        cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);
    cudaChecked(cudaMallocArray(&m_dev_tex_array, &channel_desc,
                                m_num_range_bins, max_num_pulses));

    struct cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = m_dev_tex_array;

    struct cudaTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(tex_desc));
    tex_desc.addressMode[0] = cudaAddressModeBorder;
    tex_desc.addressMode[1] = cudaAddressModeBorder;
    tex_desc.filterMode = cudaFilterModeLinear;
    tex_desc.normalizedCoords = 0;
    tex_desc.readMode = cudaReadModeElementType;

    cudaChecked(
        cudaCreateTextureObject(&m_tex_obj, &res_desc, &tex_desc, NULL));
}

SarBpGpu::~SarBpGpu() {
    FREE_AND_NULL_CUDA_DEV_ALLOC(m_dev_workbuf);
    FREE_AND_NULL_CUDA_DEVARRAY_ALLOC(m_dev_tex_array);
    cudaDestroyTextureObject(m_tex_obj);
    m_tex_obj = 0;
}

bool SarBpGpu::KernelUsesTextureMemory(SarGpuKernel kernel) {
    switch (kernel) {
    case SarGpuKernel::Invalid:
    case SarGpuKernel::DoublePrecision:
    case SarGpuKernel::MixedPrecision:
    case SarGpuKernel::IncrPhaseLookup:
    case SarGpuKernel::NewtonRaphsonTwoIter:
    case SarGpuKernel::IncrRangeSmem:
    case SarGpuKernel::SinglePrecision:
        return false;
    case SarGpuKernel::TextureSampling:
        return true;
    }
    return false;
}

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

    int num_pulses_processed = 0;
    const bool use_texture = KernelUsesTextureMemory(kernel);
    while (num_pulses_processed < num_pulses) {
        const int num_pulses_this_block =
            std::min(num_pulses, m_max_num_pulses);

        if (use_texture) {
            cudaChecked(cudaMemcpy2DToArrayAsync(
                m_dev_tex_array, 0, 0,
                dev_range_profiles + num_pulses_processed * m_num_range_bins,
                sizeof(cuComplex) * m_num_range_bins,
                sizeof(cuComplex) * m_num_range_bins, num_pulses_this_block,
                cudaMemcpyDeviceToDevice, stream));
            cudaChecked(cudaGetLastError());
        }

        SarBpGpuWrapper(dev_image, image_width, image_height, m_tex_obj,
                        dev_range_profiles +
                            num_pulses_processed * m_num_range_bins,
                        m_dev_workbuf, m_num_range_bins, num_pulses_this_block,
                        dev_ant_pos + num_pulses_processed, min_freq, dr, dx,
                        dy, 0.0f, kernel, stream);
        cudaChecked(cudaGetLastError());
        num_pulses_processed += num_pulses_this_block;
    }
}
