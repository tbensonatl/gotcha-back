#include "range_upsampling_gpu.h"

#include "helpers.h"
#include "kernels.h"

RangeUpsamplingGpu::RangeUpsamplingGpu(int num_range_bins, int num_pulses)
    : m_num_range_bins(num_range_bins), m_num_pulses(num_pulses) {
    cufftChecked(
        cufftPlan1d(&m_fft_plan, m_num_range_bins, CUFFT_C2C, m_num_pulses));
}

void RangeUpsamplingGpu::Upsample(cuComplex *dev_range_profiles,
                                  const cuComplex *host_phase_history,
                                  int num_range_freqs, cudaStream_t stream) {
    cufftChecked(cufftSetStream(m_fft_plan, stream));

    cudaChecked(cudaMemcpy2DAsync(
        dev_range_profiles, sizeof(cufftComplex) * m_num_range_bins,
        host_phase_history, sizeof(cufftComplex) * num_range_freqs,
        sizeof(cufftComplex) * num_range_freqs, m_num_pulses,
        cudaMemcpyHostToDevice, stream));

    cudaChecked(cudaMemset2DAsync(dev_range_profiles + num_range_freqs,
                                  sizeof(cuComplex) * m_num_range_bins, 0,
                                  sizeof(cuComplex) *
                                      (m_num_range_bins - num_range_freqs),
                                  m_num_pulses, stream));

    cufftChecked(cufftExecC2C(m_fft_plan, dev_range_profiles,
                              dev_range_profiles, CUFFT_INVERSE));
    FftShiftGpu(dev_range_profiles, m_num_range_bins, m_num_pulses, stream);
}

RangeUpsamplingGpu::~RangeUpsamplingGpu() {
    FREE_AND_ZERO_CUFFT_HANDLE(m_fft_plan);
}
