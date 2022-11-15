#ifndef _RANGE_UPSAMPLING_GPU_H_
#define _RANGE_UPSAMPLING_GPU_H_

#include <cstdint>

#include <cuComplex.h>
#include <cuda_runtime_api.h>
#include <cufft.h>

class RangeUpsamplingGpu {
  public:
    RangeUpsamplingGpu(int num_range_freqs, int num_range_bins, int num_pulses,
                       cudaStream_t stream);

    RangeUpsamplingGpu() = delete;
    RangeUpsamplingGpu(const RangeUpsamplingGpu &) = delete;
    RangeUpsamplingGpu &operator=(const RangeUpsamplingGpu &) = delete;

    ~RangeUpsamplingGpu();

    // phase_history should be in the frequency space with dimensions
    // num_range_freqs x num_pulses. Ideally, phase_history should be
    // a host-pinned buffer for the best performance. The returned
    // pointer to the upsampled data is owned by this class. This method
    // does not synchronize the stream, unless it does so implicitly because
    // phase_history is not a pinned buffer. Thus, the work on the stream
    // is not known to have been completed until the caller synchronizes
    // the stream.
    cuComplex *Upsample(const cuComplex *phase_history);

  private:
    int m_num_range_freqs;
    int m_num_range_bins;
    int m_num_pulses;

    cuComplex *m_dev_range_profiles{nullptr};
    cufftHandle m_fft_plan;

    // m_stream is not owned by this class
    cudaStream_t m_stream;
};

#endif // _RANGE_UPSAMPLING_GPU_H_