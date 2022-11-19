#ifndef _RANGE_UPSAMPLING_GPU_H_
#define _RANGE_UPSAMPLING_GPU_H_

#include <cstdint>

#include <cuComplex.h>
#include <cuda_runtime_api.h>
#include <cufft.h>

class RangeUpsamplingGpu {
  public:
    RangeUpsamplingGpu(int num_range_bins, int num_pulses);
    ~RangeUpsamplingGpu();

    RangeUpsamplingGpu() = delete;
    RangeUpsamplingGpu(const RangeUpsamplingGpu &) = delete;
    RangeUpsamplingGpu &operator=(const RangeUpsamplingGpu &) = delete;

    // phase_history should be in the frequency space with dimensions
    // num_range_freqs x m_num_pulses. Ideally, phase_history should be
    // a host-pinned buffer for the best performance. This method
    // does not synchronize the stream, unless it does so implicitly because
    // phase_history is not a pinned buffer. Thus, the work on the stream
    // is not known to have been completed until the caller synchronizes
    // the stream.
    void Upsample(cuComplex *dev_range_profiles,
                  const cuComplex *host_phase_history, int num_range_freqs,
                  cudaStream_t stream);

  private:
    int m_num_range_bins;
    int m_num_pulses;

    cufftHandle m_fft_plan;
};

#endif // _RANGE_UPSAMPLING_GPU_H_