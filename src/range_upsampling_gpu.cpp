#include "range_upsampling_gpu.h"

#include "helpers.h"
#include "kernels.h"

RangeUpsamplingGpu::RangeUpsamplingGpu(int num_range_freqs, int num_range_bins,
                                       int num_pulses, cudaStream_t stream)
    : m_num_range_freqs(num_range_freqs), m_num_range_bins(num_range_bins),
      m_num_pulses(num_pulses), m_stream(stream) {
    cudaChecked(
        cudaMalloc((void **)&m_dev_range_profiles,
                   m_num_pulses * m_num_range_bins * sizeof(cuComplex)));
    cufftChecked(
        cufftPlan1d(&m_fft_plan, m_num_range_bins, CUFFT_C2C, m_num_pulses));
    cufftChecked(cufftSetStream(m_fft_plan, m_stream));
}

cuComplex *RangeUpsamplingGpu::Upsample(const cuComplex *phase_history) {
    cudaChecked(cudaMemcpy2DAsync(
        m_dev_range_profiles, sizeof(cufftComplex) * m_num_range_bins,
        phase_history, sizeof(cufftComplex) * m_num_range_freqs,
        sizeof(cufftComplex) * m_num_range_freqs, m_num_pulses,
        cudaMemcpyHostToDevice, m_stream));

    cudaChecked(cudaMemset2DAsync(m_dev_range_profiles + m_num_range_freqs,
                                  sizeof(cuComplex) * m_num_range_bins, 0,
                                  sizeof(cuComplex) *
                                      (m_num_range_bins - m_num_range_freqs),
                                  m_num_pulses, m_stream));

    cufftChecked(cufftExecC2C(m_fft_plan, m_dev_range_profiles,
                              m_dev_range_profiles, CUFFT_INVERSE));
    FftShiftGpu(m_dev_range_profiles, m_num_range_bins, m_num_pulses, m_stream);

    return m_dev_range_profiles;
}

RangeUpsamplingGpu::~RangeUpsamplingGpu() {
    // This class does not own m_stream, so does not destroy it
    FREE_AND_NULL_CUDA_DEV_ALLOC(m_dev_range_profiles);
    FREE_AND_ZERO_CUFFT_HANDLE(m_fft_plan);
}
