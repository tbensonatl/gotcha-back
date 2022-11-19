#include <cstring>
#include <filesystem>
#include <mutex>
#include <vector>

#include <cuda_runtime.h>

#include "data_reader.h"
#include "helpers.h"
#include "kernels.h"
#include "video_sar.h"

namespace fs = std::filesystem;

VideoSar::VideoSar(const VideoParams &params, int device_id, SarUI &ui)
    : m_ui(ui), m_params(params) {
    int num_devices = -1;
    cudaChecked(cudaGetDeviceCount(&num_devices));
    if (device_id < 0 || device_id >= num_devices) {
        LOG_ERR("Invalid GPU device ID %d; have %d available devices.",
                device_id, num_devices);
        exit(EXIT_FAILURE);
    }

    cudaChecked(cudaSetDevice(device_id));

    InitCudaBuffers();
}

void VideoSar::FreeCudaBuffers() {
    cudaStreamDestroy(m_stream);
    cudaFreeHost(m_pinned.ant_pos);
    cudaFreeHost(m_pinned.image);
    cudaFreeHost(m_pinned.phase_history);
    cudaFreeHost(m_pinned.display_image);
    cudaFree(m_dev.max_magnitude_workbuf);
    cudaFree(m_dev.ant_pos);
    cudaFree(m_dev.image);
    cudaFree(m_dev.magnitude_image);
    cudaFree(m_dev.resampled_magnitude_image);
    cudaFree(m_dev.range_profiles);
}

void VideoSar::InitCudaBuffers() {
    const int pass = 1;
    const Gotcha::DataRequest request = {
        .first_az = 1,
        .last_az = 360,
        .pass = pass,
        .polarization = Gotcha::Polarization::HH,
    };

    Gotcha::Reader reader(m_params.gotcha_dir);
    const SarReturnCode rc = reader.ReadDataSet(m_data_set, request);
    if (rc != SarReturnCode::Success) {
        LOG_ERR("Failed to read data set: %s", GetSarReturnCodeString(rc));
        exit(EXIT_FAILURE);
    }

    m_num_upsampled_bins = static_cast<int>(
        pow(2, ceil(log(m_params.range_upsample_factor * m_data_set.num_freq) /
                    log(2))));

    const int n_pulses = m_data_set.num_pulses;
    const int n_freq = m_data_set.num_freq;
    const int nx = m_params.image_width;
    const int ny = m_params.image_height;
    const int max_nx = m_params.max_screen_width;
    const int max_ny = m_params.max_screen_height;

    LOG("Loaded %d pulses", n_pulses);

    cudaChecked(
        cudaMalloc((void **)&m_dev.range_profiles,
                   n_pulses * m_num_upsampled_bins * sizeof(cuComplex)));
    cudaChecked(cudaMalloc((void **)&m_dev.image, nx * ny * sizeof(cuComplex)));
    cudaChecked(cudaMalloc((void **)&m_dev.ant_pos, sizeof(float3) * n_pulses));
    const size_t mag_workbuf_size_bytes = GetMaxMagnitudeWorkBufSizeBytes(ny);
    cudaChecked(cudaMalloc((void **)&m_dev.max_magnitude_workbuf,
                           mag_workbuf_size_bytes));
    cudaChecked(cudaMalloc((void **)&m_dev.magnitude_image,
                           sizeof(uint32_t) * nx * ny));
    cudaChecked(cudaMalloc((void **)&m_dev.resampled_magnitude_image,
                           sizeof(uint32_t) * max_nx * max_ny));

    cudaChecked(cudaMallocHost((void **)&m_pinned.phase_history,
                               n_pulses * n_freq * sizeof(cuComplex)));
    cudaChecked(
        cudaMallocHost((void **)&m_pinned.image, nx * ny * sizeof(cuComplex)));
    cudaChecked(cudaMallocHost((void **)&m_pinned.ant_pos,
                               m_data_set.num_pulses * sizeof(float3)));
    cudaChecked(cudaMallocHost((void **)&m_pinned.display_image,
                               sizeof(uint32_t) * max_nx * max_ny));

    memcpy(m_pinned.phase_history, &m_data_set.phase_history[0],
           sizeof(cuComplex) * n_pulses * n_freq);
    memcpy(m_pinned.ant_pos, &m_data_set.ant_pos[0], sizeof(float3) * n_pulses);

    cudaChecked(cudaStreamCreate(&m_stream));

    m_bp.reset(new SarBpGpu(m_num_upsampled_bins));
    m_range_upsampling.reset(
        new RangeUpsamplingGpu(m_num_upsampled_bins, n_pulses));
}

void VideoSar::Render(int screen_width, int screen_height) {
    static bool have_copied_data = false;

    const int n_pulses = m_data_set.num_pulses;
    const int n_freq = m_data_set.num_freq;
    const int az_degrees_per_image = m_params.az_degrees_per_image;

    const int n_pulses_per_az_deg = static_cast<int>(roundf(n_pulses / 360.0f));
    const int first_pulse_this_img =
        m_start_az_next_image * n_pulses_per_az_deg;
    const int n_pulses_this_img = n_pulses_per_az_deg * az_degrees_per_image;

    const int nx = m_params.image_width;
    const int ny = m_params.image_height;

    if (!have_copied_data) {
        cudaChecked(cudaMemcpyAsync(m_dev.ant_pos, m_pinned.ant_pos,
                                    sizeof(float3) * n_pulses,
                                    cudaMemcpyHostToDevice, m_stream));
        m_range_upsampling->Upsample(m_dev.range_profiles,
                                     m_pinned.phase_history, n_freq, m_stream);
        have_copied_data = true;
    }

    cudaChecked(
        cudaMemsetAsync(m_dev.image, 0, sizeof(cuComplex) * nx * ny, m_stream));

    const float image_field_of_view_m = 150.0f;

    // If this image require pulses that wrap around back to 0 in azimuth,
    // then split the backprojection into two parts
    const int n_pulses_this_bp =
        (first_pulse_this_img + n_pulses_this_img <= n_pulses)
            ? n_pulses_this_img
            : n_pulses - first_pulse_this_img;
    m_bp->Backproject(
        m_dev.image,
        m_dev.range_profiles + first_pulse_this_img * m_num_upsampled_bins,
        m_dev.ant_pos + first_pulse_this_img, nx, ny, n_pulses_this_bp,
        image_field_of_view_m, m_data_set.min_freq, m_data_set.del_freq,
        m_params.kernel, m_stream);
    const int pulses_remaining = n_pulses_this_img - n_pulses_this_bp;
    if (pulses_remaining > 0) {
        m_bp->Backproject(m_dev.image, m_dev.range_profiles, m_dev.ant_pos, nx,
                          ny, pulses_remaining, image_field_of_view_m,
                          m_data_set.min_freq, m_data_set.del_freq,
                          m_params.kernel, m_stream);
    }

    const float min_normalized_db = -70;
    ComputeMagnitudeImage(m_dev.magnitude_image, m_dev.max_magnitude_workbuf,
                          m_dev.image, nx, ny, min_normalized_db, m_stream);

    ResampleMagnitudeImage(m_dev.resampled_magnitude_image,
                           m_dev.magnitude_image, screen_width, screen_height,
                           nx, ny, m_stream);

    cudaChecked(cudaMemcpyAsync(m_pinned.display_image,
                                m_dev.resampled_magnitude_image,
                                sizeof(uint32_t) * screen_width * screen_height,
                                cudaMemcpyDeviceToHost, m_stream));

    cudaChecked(cudaStreamSynchronize(m_stream));

    cudaChecked(cudaDeviceSynchronize());

    m_start_az_next_image =
        (m_start_az_next_image + az_degrees_per_image) % 360;
}

void VideoSar::Run() {
    while (1) {
        if (m_stop) {
            break;
        }
        m_params.kernel = m_ui.GetSelectedKernel();
        std::pair<int, int> wh = m_ui.GetImageScreenDimensions();
        Render(wh.first, wh.second);
        if (m_stop) {
            break;
        }
        m_ui.UpdateImage(m_pinned.display_image, wh.first, wh.second);
    }
}

void VideoSar::Stop() { m_stop = true; }