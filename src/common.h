#ifndef _COMMON_H_
#define _COMMON_H_

#include <string>

#define SPEED_OF_LIGHT_M_PER_SEC (299792458.0)

enum class SarGpuKernel {
    Invalid = 0,
    DoublePrecision,
    MixedPrecision,
    SmemRange,
    IncrPhaseLookup,
    SinglePrecision
};

enum class SarReturnCode { Success = 0, InvalidArg, DataReadError };

const char *GetSarReturnCodeString(SarReturnCode code);

struct ReconParams {
    // Directory containing unpacked Gotcha data
    std::string gotcha_dir;
    // Image width in pixels
    int image_width{1024};
    // Image height in pixels
    int image_height{1024};
    // Image field of view (meters)
    float image_field_of_view_m{125.0f};
    // Phase history upsampling factor
    int range_upsample_factor{16};
    // First azimuth angle in degrees
    int first_az{1};
    // Last azimuth angle in degrees
    int last_az{5};
    // Platform pass number of data
    int pass{1};
    // Selected kernel for GPU-based recons
    SarGpuKernel kernel{SarGpuKernel::DoublePrecision};
};

struct VideoParams {
    // Directory containing unpacked Gotcha data
    std::string gotcha_dir;
    // Image width in pixels
    int image_width{1024};
    // Image height in pixels
    int image_height{1024};
    // Maximum screen width in pixels
    int max_screen_width{2048};
    // Maximum screen height in pixels
    int max_screen_height{2048};
    // Image field of view (meters)
    float image_field_of_view_m{125.0f};
    // Phase history upsampling factor
    int range_upsample_factor{16};
    // Azimuth degrees per image. Images will be reconstructed in succession
    // using this many degrees per image
    int az_degrees_per_image{1};
    // Platform pass number of data
    int pass{1};
    // Selected kernel for GPU-based recons
    SarGpuKernel kernel{SarGpuKernel::DoublePrecision};
};

SarGpuKernel IntToSarGpuKernel(int kernel);

#endif // _COMMON_H_