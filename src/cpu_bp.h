#include "data_reader.h"

#include <cuComplex.h>

void cpu_sar_bp(std::vector<cuComplex> &image, int image_width,
                int image_height, const std::vector<cuComplex> &range_profiles,
                int num_range_bins, int num_pulses,
                const std::vector<float3> &ant_pos,
                double phase_correction_partial, double dr, double dx,
                double dy, double z0);