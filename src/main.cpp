#include <chrono>
#include <cinttypes>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>

#include <boost/program_options.hpp>

#include <cuda_runtime_api.h>
#include <cufft.h>

#include "common.h"
#include "cpu_bp.h"
#include "data_reader.h"
#include "fft.h"
#include "helpers.h"
#include "kernels.h"
#include "ser.h"
#include "video_sar.h"

#include "sar_ui.h"

namespace po = boost::program_options;
namespace fs = std::filesystem;

static int GetNumUpsampledRangeBins(int num_freqs, int upsample_factor) {
    return static_cast<int>(
        pow(2, ceil(log(upsample_factor * num_freqs) / log(2))));
}

static void runCpu(std::vector<cuComplex> &image, const ReconParams &params) {
    Gotcha::DataBlock data_set;
    const Gotcha::DataRequest request = {
        .first_az = params.first_az,
        .last_az = params.last_az,
        .pass = params.pass,
        .polarization = Gotcha::Polarization::HH,
    };

    SarReturnCode rc = SarReturnCode::Success;

    Gotcha::Reader reader(params.gotcha_dir);
    rc = reader.ReadDataSet(data_set, request);
    if (rc != SarReturnCode::Success) {
        LOG_ERR("Failed to read data set: %s", GetSarReturnCodeString(rc));
        exit(EXIT_FAILURE);
    }

    const int Nfft = GetNumUpsampledRangeBins(data_set.num_freq,
                                              params.range_upsample_factor);

    std::vector<cuComplex> range_profiles(data_set.num_pulses * Nfft);
    std::vector<cuComplex> workbuf(Nfft);
    FftRadix2 fft;
    for (int i = 0; i < data_set.num_pulses; i++) {
        memcpy(&workbuf[0], &data_set.phase_history[i * data_set.num_freq],
               sizeof(cuComplex) * data_set.num_freq);
        if (Nfft > data_set.num_freq) {
            memset(&workbuf[data_set.num_freq], 0,
                   sizeof(cuComplex) * (Nfft - data_set.num_freq));
        }

        rc = fft.Transform(workbuf, true);
        if (rc != SarReturnCode::Success) {
            LOG_ERR("FFT tranform failed: %s", GetSarReturnCodeString(rc));
            exit(EXIT_FAILURE);
        }

        for (int k = 0; k < Nfft / 2; k++) {
            // swap halves for a in-place fftshift (i.e. to center DC)
            range_profiles[i * Nfft + k] = workbuf[Nfft / 2 + k];
        }
        for (int k = 0; k < Nfft / 2; k++) {
            // swap halves for a in-place fftshift (i.e. to center DC)
            range_profiles[i * Nfft + Nfft / 2 + k] = workbuf[k];
        }
    }

    const double c = SPEED_OF_LIGHT_M_PER_SEC;
    const double maxWr = c / (2.0 * data_set.del_freq);

    data_set.del_r = maxWr / Nfft;

    const int Nx = params.image_width;
    const int Ny = params.image_height;
    const double dx = params.image_field_of_view_m / Nx;
    const double dy = params.image_field_of_view_m / Ny;

    const double dR = maxWr / Nfft;

    // The data has been motion-compenstated so that the origin corresponds
    // to DC. The phase compenstation for a pixel with distance delR from
    // the origin is thus:
    //   exp(1i * 4 * pi * minum_freq * delR / c)
    // Note that delR = 0 yields a value of 1 (no phase correction).
    // Everything except delR is fixed, so we precompute it.
    const double phase_correction_partial =
        4.0 * M_PI * (data_set.min_freq / c);

    cpu_sar_bp(image, Nx, Ny, range_profiles, Nfft, data_set.num_pulses,
               data_set.ant_pos, phase_correction_partial, dR, dx, dy, 0.0f);
}

static void runGpu(std::vector<cuComplex> &image, const ReconParams &params,
                   int device_id) {
    Gotcha::DataBlock data_set;
    const Gotcha::DataRequest request = {
        .first_az = params.first_az,
        .last_az = params.last_az,
        .pass = params.pass,
        .polarization = Gotcha::Polarization::HH,
    };

    int num_devices = 0;
    cudaChecked(cudaGetDeviceCount(&num_devices));
    if (device_id < 0 || device_id >= num_devices) {
        LOG_ERR("Invalid GPU device ID %d; have %d available devices.",
                device_id, num_devices);
        exit(EXIT_FAILURE);
    }

    cudaChecked(cudaSetDevice(device_id));

    Gotcha::Reader reader(params.gotcha_dir);
    const SarReturnCode rc = reader.ReadDataSet(data_set, request);
    if (rc != SarReturnCode::Success) {
        LOG_ERR("Failed to read input data: %s", GetSarReturnCodeString(rc));
        exit(EXIT_FAILURE);
    }

    const int Nx = params.image_width;
    const int Ny = params.image_height;
    const int num_range_bins = GetNumUpsampledRangeBins(
        data_set.num_freq, params.range_upsample_factor);
    LOG("Read %d pulses with %d frequencies per pulse. Upsampling to %d range "
        "bins.",
        data_set.num_pulses, data_set.num_freq, num_range_bins);

    cuComplex *dev_range_profiles = nullptr, *dev_image = nullptr;
    cudaChecked(
        cudaMalloc((void **)&dev_range_profiles,
                   data_set.num_pulses * num_range_bins * sizeof(cuComplex)));
    cudaChecked(cudaMalloc((void **)&dev_image, Nx * Ny * sizeof(cuComplex)));
    float3 *dev_ant_pos = nullptr;
    cudaChecked(cudaMalloc((void **)&dev_ant_pos,
                           sizeof(float3) * data_set.num_pulses));

    const size_t bp_workbuf_bytes =
        GetMaxBackprojWorkBufSizeBytes(num_range_bins);
    uint8_t *dev_bp_workbuf = nullptr;
    if (bp_workbuf_bytes > 0) {
        cudaChecked(cudaMalloc((void **)&dev_bp_workbuf, bp_workbuf_bytes));
    }

    const int num_pulses = data_set.num_pulses;
    const int num_freq = data_set.num_freq;
    cuComplex *phase_history = nullptr, *pinned_image = nullptr,
              *ant_pos = nullptr;
    cudaChecked(cudaMallocHost((void **)&phase_history,
                               num_pulses * num_freq * sizeof(cuComplex)));
    cudaChecked(
        cudaMallocHost((void **)&pinned_image, Nx * Ny * sizeof(cuComplex)));
    cudaChecked(cudaMallocHost((void **)&ant_pos, num_pulses * sizeof(float3)));

    memcpy(phase_history, &data_set.phase_history[0],
           sizeof(cuComplex) * num_pulses * num_freq);
    memcpy(ant_pos, &data_set.ant_pos[0], sizeof(float3) * num_pulses);

    cudaStream_t stream;
    cufftHandle fft_plan;

    cudaChecked(cudaStreamCreate(&stream));
    cufftChecked(
        cufftPlan1d(&fft_plan, num_range_bins, CUFFT_C2C, data_set.num_pulses));
    cufftChecked(cufftSetStream(fft_plan, stream));

    cudaChecked(cudaMemcpyAsync(dev_ant_pos, ant_pos,
                                sizeof(float3) * data_set.num_pulses,
                                cudaMemcpyHostToDevice, stream));
    cudaChecked(cudaMemcpy2DAsync(
        dev_range_profiles, sizeof(cufftComplex) * num_range_bins,
        phase_history, sizeof(cufftComplex) * data_set.num_freq,
        sizeof(cufftComplex) * data_set.num_freq, data_set.num_pulses,
        cudaMemcpyHostToDevice, stream));
    cudaChecked(cudaMemset2DAsync(dev_range_profiles + data_set.num_freq,
                                  sizeof(cuComplex) * num_range_bins, 0,
                                  sizeof(cuComplex) *
                                      (num_range_bins - data_set.num_freq),
                                  data_set.num_pulses, stream));
    cudaChecked(
        cudaMemsetAsync(dev_image, 0, sizeof(cuComplex) * Nx * Ny, stream));

    cufftChecked(cufftExecC2C(fft_plan, dev_range_profiles, dev_range_profiles,
                              CUFFT_INVERSE));
    FftShiftGpu(dev_range_profiles, num_range_bins, data_set.num_pulses,
                stream);

    const double c = SPEED_OF_LIGHT_M_PER_SEC;
    const double max_wr = c / (2.0 * data_set.del_freq);
    const double dx = params.image_field_of_view_m / Nx;
    const double dy = params.image_field_of_view_m / Ny;
    const double dr = max_wr / num_range_bins;

    cudaChecked(cudaStreamSynchronize(stream));
    const auto t1 = std::chrono::steady_clock::now();

    SarBpGpu(dev_image, Nx, Ny, dev_range_profiles, dev_bp_workbuf,
             num_range_bins, data_set.num_pulses, dev_ant_pos,
             data_set.min_freq, dr, dx, dy, 0.0f, params.kernel, stream);

    cudaChecked(cudaStreamSynchronize(stream));
    const auto t2 = std::chrono::steady_clock::now();
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    const double gbps = (((double)data_set.num_pulses * Nx * Ny) / 1.0e9) /
                        ((double)elapsed / 1.0e6);
    LOG("Elapsed time (kern=%d):\t\t\t%" PRId64 " microseconds", params.kernel,
        elapsed);
    LOG("Giga backprojections per second:\t%.2f", gbps);

    cudaChecked(cudaMemcpyAsync(pinned_image, dev_image,
                                sizeof(cuComplex) * Nx * Ny,
                                cudaMemcpyHostToDevice, stream));
    cudaChecked(cudaStreamSynchronize(stream));

    memcpy(&image[0], pinned_image, sizeof(cuComplex) * Nx * Ny);

    cufftDestroy(fft_plan);
    cudaStreamDestroy(stream);
    if (dev_bp_workbuf != nullptr) {
        cudaFree(dev_bp_workbuf);
    }
    cudaFree(dev_ant_pos);
    cudaFree(dev_image);
    cudaFree(dev_range_profiles);
    cudaFreeHost(ant_pos);
    cudaFreeHost(pinned_image);
    cudaFreeHost(phase_history);
}

static void readGoldImage(std::vector<cuComplex> &image,
                          const std::string &filename) {
    fs::path input = filename;
    const auto size_bytes = fs::file_size(input);

    int num_pixels = size_bytes / sizeof(cuComplex);
    image.resize(num_pixels);
    std::ifstream ifs(filename, std::ios::binary | std::ios::in);
    ifs.read(reinterpret_cast<char *>(&image[0]),
             sizeof(cuComplex) * num_pixels);
    const bool failed = ifs.fail();
    ifs.close();
    if (failed) {
        throw std::runtime_error("failed to read gold input image");
    }
}

void runUI(SarUI &ui) { ui.StartEventLoop(); }

int main(int argc, char **argv) {
    po::options_description desc("Allowed options");
    desc.add_options()("help,h", "help message")(
        "first-az", po::value<int>()->default_value(38),
        "First azimuthal angle to use for reconstruction")(
        "last-az", po::value<int>()->default_value(41),
        "Last azimuthal angle to use for reconstruction")(
        "kern", po::value<int>()->default_value(1),
        "GPU kernel to use for backprojection")(
        "pass", po::value<int>()->default_value(1),
        "Data pass to use for reconstruction")(
        "gotcha-dir", po::value<std::string>(),
        "Path to GOTCHA directory containing GOTCHA-CP_Disc1 subdirectory")(
        "gpu", po::value<int>(),
        "Enables GPU-based backprojection on the specified device")(
        "ser", po::value<std::string>(),
        "Compute signal-to-error db metric against reference image")(
        "video", po::value<int>(),
        "Enables video mode using specified device to render data to the GUI")(
        "output-file,o", po::value<std::string>()->default_value("image.bin"),
        "Output filename for reconstructed image");
    po::positional_options_description pos;
    pos.add("gotcha-dir", 1);

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv)
                      .options(desc)
                      .positional(pos)
                      .run(),
                  vm);
        po::notify(vm);
    } catch (po::error &e) {
        std::cout << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    if (argc == 1 || vm.count("help")) {
        std::cout << "Usage: " << argv[0] << " [options] <gotcha-dir>\n";
        std::cout << desc;
        return EXIT_SUCCESS;
    }

    if (vm.count("gotcha-dir") == 0) {
        std::cout << "Usage: " << argv[0] << " [options] <gotcha-dir>\n";
        std::cout << desc;
        return EXIT_FAILURE;
    }

    fs::path gotcha_dir =
        vm["gotcha-dir"].as<std::string>() + "/GOTCHA-CP_Disc1/DATA";
    if (!fs::is_directory(gotcha_dir)) {
        std::cerr << gotcha_dir << " is not a valid directory" << std::endl;
        return EXIT_FAILURE;
    }

    const bool useGPU = vm.count("gpu") || vm.count("video");
    const SarGpuKernel selected_kernel =
        IntToSarGpuKernel(vm["kern"].as<int>());
    if (useGPU && selected_kernel == SarGpuKernel::Invalid) {
        LOG_ERR("Invalid GPU kernel selection %d", vm["kern"].as<int>());
        exit(EXIT_FAILURE);
    }

    const int first_az = vm["first-az"].as<int>();
    const int last_az = vm["last-az"].as<int>();
    const int pass = vm["pass"].as<int>();

    const ReconParams recon_params = {
        .gotcha_dir = gotcha_dir,
        .image_width = 1024,
        .image_height = 1024,
        .first_az = first_az,
        .last_az = last_az,
        .pass = pass,
        .kernel = selected_kernel,
    };

    const VideoParams video_params = {
        .gotcha_dir = gotcha_dir,
        .image_width = 1024,
        .image_height = 1024,
        .az_degrees_per_image = 10,
        .pass = pass,
        .kernel = selected_kernel,
    };
    const int num_pixels = recon_params.image_width * recon_params.image_height;
    assert(num_pixels > 0);

    std::vector<cuComplex> ser_gold_image;
    if (vm.count("ser") > 0) {
        const std::string gold_filename = vm["ser"].as<std::string>();
        try {
            readGoldImage(ser_gold_image, gold_filename);
        } catch (const std::exception &e) {
            std::cerr << e.what() << std::endl;
            return EXIT_FAILURE;
        } catch (...) {
            std::cerr << "Caught unknown exception" << std::endl;
            return EXIT_FAILURE;
        }
        if (ser_gold_image.size() != static_cast<size_t>(num_pixels)) {
            LOG_ERR(
                "Incompatible size for gold image: got %zu pixels, expected %d",
                ser_gold_image.size(), num_pixels);
            return EXIT_FAILURE;
        }
        LOG("Read gold image from %s [%d pixels]", gold_filename.c_str(),
            num_pixels);
    }

    std::vector<cuComplex> image(recon_params.image_width *
                                 recon_params.image_height);

    try {
        if (vm.count("video")) {
            const int WINDOW_INITIAL_WIDTH = 512;
            const int WINDOW_INITIAL_HEIGHT = 512;
            SarUI ui(WINDOW_INITIAL_WIDTH, WINDOW_INITIAL_HEIGHT);
            std::thread ui_thread(&SarUI::StartEventLoop, &ui);
            VideoSar sar(video_params, vm["video"].as<int>(), ui);
            std::thread sar_thread(&VideoSar::Run, &sar);
            ui_thread.join();
            sar.Stop();
            sar_thread.join();
        } else if (vm.count("gpu")) {
            runGpu(image, recon_params, vm["gpu"].as<int>());
        } else {
            runCpu(image, recon_params);
        }
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    } catch (...) {
        std::cerr << "Caught unknown exception" << std::endl;
        return EXIT_FAILURE;
    }

    int exit_code = EXIT_SUCCESS;

    if (vm.count("ser")) {
        const double ser = signal_to_error_ratio(ser_gold_image, image);
        LOG("Signal-to-error ratio:\t\t\t%.2f dB", ser);
    }

    if (!vm.count("video")) {
        fs::path output_file = vm["output-file"].as<std::string>();
        std::ofstream outfile(output_file.string().c_str(),
                              std::ios::out | std::ios::binary);
        LOG("Writing output to %s", output_file.string().c_str());
        if (outfile.is_open()) {
            outfile.write(reinterpret_cast<const char *>(&image[0]),
                          sizeof(cuComplex) * num_pixels);
            if (outfile.bad()) {
                LOG_ERR("Failed to write image to %s",
                        output_file.string().c_str());
                exit_code = EXIT_FAILURE;
            }
            outfile.close();
        } else {
            LOG_ERR("Failed to open %s for writing",
                    output_file.string().c_str());
            exit_code = EXIT_FAILURE;
        }
    }

    return exit_code;
}