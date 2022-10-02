#include "cpu_bp.h"

void cpu_sar_bp(std::vector<cuComplex> &image, int image_width,
                int image_height, const std::vector<cuComplex> &range_profiles,
                int num_range_bins, int num_pulses,
                const std::vector<float3> &ant_pos,
                double phase_correction_partial, double dr, double dx,
                double dy, double z0) {
    int p, ix, iy;

    const double dr_inv = 1.0 / dr;
    for (iy = 0; iy < image_height; ++iy) {
        const double py = (-image_height / 2.0 + 0.5 + iy) * dy;
        for (ix = 0; ix < image_width; ++ix) {
            cuDoubleComplex accum = make_cuDoubleComplex(0.0, 0.0);
            const double px = (-image_width / 2.0 + 0.5 + ix) * dx;

            for (p = 0; p < num_pulses; ++p) {
                const int pulse_offset = p * num_range_bins;

                /* calculate the range R from the antenna to this pixel */
                const double ax = ant_pos[p].x;
                const double ay = ant_pos[p].y;
                const double az = ant_pos[p].z;
                const double xdiff = ax - px;
                const double ydiff = ay - py;
                const double zdiff = az - z0;
                const double aw = sqrt(ax * ax + ay * ay + az * az);

                /* differential range to the motion compenstation point (i.e.
                 * origin) */
                const double diff_r =
                    sqrt(xdiff * xdiff + ydiff * ydiff + zdiff * zdiff) - aw;
                const double bin = diff_r * dr_inv + num_range_bins / 2.0;
                if (bin >= 0.0 && bin < num_range_bins - 2.0) {
                    /* interpolation range is [bin_floor, bin_floor+1)  */
                    const int bin_floor = (int)bin;

                    /* interpolation weight */
                    const double w = (bin - (double)bin_floor);

                    /* linearly interpolate to obtain a sample at bin */
                    const int ind_base = pulse_offset + bin_floor;
                    const cuDoubleComplex sample = make_cuDoubleComplex(
                        (1.0 - w) * range_profiles[ind_base].x +
                            w * range_profiles[ind_base + 1].x,
                        (1.0 - w) * range_profiles[ind_base].y +
                            w * range_profiles[ind_base + 1].y);

                    double sinx, cosx;
                    sincos(phase_correction_partial * diff_r, &sinx, &cosx);
                    /* compute the complex exponential for the matched filter */
                    const cuDoubleComplex matched_filter =
                        make_cuDoubleComplex(cosx, sinx);

                    /* accumulate this pulse's contribution into the pixel */
                    accum = cuCfma(sample, matched_filter, accum);
                }
            }

            image[iy * image_width + ix] = make_cuComplex(accum.x, accum.y);
        }
    }
}
