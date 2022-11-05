#ifndef _VIDEO_SAR_
#define _VIDEO_SAR_

#include <filesystem>

#include <cuda_runtime.h>

#include "common.h"
#include "data_reader.h"
#include "sar_ui.h"

class VideoSar {
  public:
    VideoSar(const VideoParams &params, int device_id, SarUI &ui);
    ~VideoSar() {
        FreeCudaBuffers();
        m_data_set.Reset();
    }

    void Run();
    void Stop();

  private:
    SarUI &m_ui;
    SarGpuKernel m_kernel;
    VideoParams m_params;
    Gotcha::DataBlock m_data_set;

    int m_num_upsampled_bins{0};
    int m_start_az_next_image{0};

    bool m_stop{false};

    // Device buffers
    struct {
        cuComplex *range_profiles{nullptr};
        cuComplex *image{nullptr};
        float3 *ant_pos{nullptr};
        float *max_magnitude_workbuf{nullptr};
        uint8_t *bp_workbuf{nullptr};
        uint32_t *magnitude_image{nullptr};
        uint32_t *resampled_magnitude_image{nullptr};
    } m_dev;

    // Host pinned buffers
    struct {
        cuComplex *phase_history{nullptr};
        cuComplex *image{nullptr};
        cuComplex *ant_pos{nullptr};
        uint32_t *display_image{nullptr};
    } m_pinned;

    cudaStream_t m_stream;
    cufftHandle m_fft_plan;

    void InitCudaBuffers();
    void FreeCudaBuffers();
    void Render(int screen_width, int screen_height);
};

#endif // _VIDEO_SAR_
