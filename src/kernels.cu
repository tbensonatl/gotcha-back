#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "helpers.h"
#include "kernels.h"

namespace cg = cooperative_groups;

static inline int idivup(int x, int y) { return (x + y - 1) / y; }

__global__ void FftShiftKernel(cuComplex *data, int num_samples,
                               int num_arrays) {
    const int M = blockIdx.x;
    if (M >= num_arrays) {
        return;
    }
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int offset = M * num_samples;
    for (int t = tid; t < num_samples / 2; t += nthreads) {
        const cuComplex tmp = data[offset + t];
        data[offset + t] = data[offset + t + num_samples / 2];
        data[offset + t + num_samples / 2] = tmp;
    }
}

__global__ void
SarBpKernelDoublePrecision(cuComplex *image, int image_width, int image_height,
                           const cuComplex *range_profiles, int num_range_bins,
                           int num_pulses, const float3 *ant_pos,
                           double phase_correction_partial, double dr_inv,
                           double dx, double dy, double z0) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= image_width || iy >= image_height) {
        return;
    }

    const double py = (image_height / 2.0 - 0.5 - iy) * dy;
    const double px = (-image_width / 2.0 + 0.5 + ix) * dx;

    cuDoubleComplex accum = make_cuDoubleComplex(0.0, 0.0);
    const double bin_offset = 0.5 * num_range_bins;
    const double max_bin_f = num_range_bins - 2.0f;
    // See the reference CPU version for comments on the backprojection
    // operations
    for (int p = 0; p < num_pulses; ++p) {
        const int pulse_offset = p * num_range_bins;
        const float3 ap = ant_pos[p];
        const double apx = ap.x;
        const double apy = ap.y;
        const double apz = ap.z;
        const double range_to_center = sqrt(apx * apx + apy * apy + apz * apz);
        const double xdiff = apx - px;
        const double ydiff = apy - py;
        const double zdiff = apz - z0;
        const double diffR =
            sqrt(xdiff * xdiff + ydiff * ydiff + zdiff * zdiff) -
            range_to_center;

        const double bin = diffR * dr_inv + bin_offset;
        if (bin >= 0.0f && bin < max_bin_f) {
            const int bin_floor = (int)bin;
            const double w = (bin - (double)bin_floor);
            const int ind_base = pulse_offset + bin_floor;
            const cuDoubleComplex sample =
                make_cuDoubleComplex((1.0 - w) * range_profiles[ind_base].x +
                                         w * range_profiles[ind_base + 1].x,
                                     (1.0 - w) * range_profiles[ind_base].y +
                                         w * range_profiles[ind_base + 1].y);

            double sinx, cosx;
            sincos(phase_correction_partial * diffR, &sinx, &cosx);
            const cuDoubleComplex matched_filter =
                make_cuDoubleComplex(cosx, sinx);

            accum = cuCfma(sample, matched_filter, accum);
        }
    }

    const int img_ind = iy * image_width + ix;
    image[img_ind].x += (float)accum.x;
    image[img_ind].y += (float)accum.y;
}

__global__ void
SarBpKernelMixedPrecision(cuComplex *image, int image_width, int image_height,
                          const cuComplex *range_profiles, int num_range_bins,
                          int num_pulses, const float3 *ant_pos,
                          double phase_correction_partial, double dr_inv,
                          double dx, double dy, double z0) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= image_width || iy >= image_height) {
        return;
    }

    const double py = (image_height / 2.0 - 0.5 - iy) * dy;
    const double px = (-image_width / 2.0 + 0.5 + ix) * dx;

    cuComplex accum = make_cuComplex(0.0, 0.0);
    const float bin_offset = 0.5f * num_range_bins;
    const float max_bin_f = num_range_bins - 2.0f;
    for (int p = 0; p < num_pulses; ++p) {
        const int pulse_offset = p * num_range_bins;
        const float3 ap = ant_pos[p];
        const double apx = ap.x;
        const double apy = ap.y;
        const double apz = ap.z;
        const double range_to_center = sqrt(apx * apx + apy * apy + apz * apz);
        const double xdiff = apx - px;
        const double ydiff = apy - py;
        const double zdiff = apz - z0;
        const double diffR =
            sqrt(xdiff * xdiff + ydiff * ydiff + zdiff * zdiff) -
            range_to_center;

        const float bin = (float)(diffR * dr_inv) + bin_offset;
        if (bin >= 0.0f && bin < max_bin_f) {
            const int bin_floor = (int)bin;
            const float w = (bin - (float)bin_floor);
            const int ind_base = pulse_offset + bin_floor;
            const cuComplex sample =
                make_cuComplex((1.0f - w) * range_profiles[ind_base].x +
                                   w * range_profiles[ind_base + 1].x,
                               (1.0f - w) * range_profiles[ind_base].y +
                                   w * range_profiles[ind_base + 1].y);

            double sinx, cosx;
            sincos(phase_correction_partial * diffR, &sinx, &cosx);
            const cuComplex matched_filter = make_cuComplex(cosx, sinx);
            accum = cuCfmaf(sample, matched_filter, accum);
        }
    }

    const int img_ind = iy * image_width + ix;
    image[img_ind] = cuCaddf(image[img_ind], accum);
}

__global__ void
SarBpKernelSmemRange(cuComplex *image, int image_width, int image_height,
                     const cuComplex *range_profiles, int num_range_bins,
                     int num_pulses, const float3 *ant_pos,
                     double phase_correction_partial, double dr_inv, double dx,
                     double dy, double z0) {
    extern __shared__ double smem_dists[];

    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    cg::thread_block block = cg::this_thread_block();

    const double xmid = (-image_width / 2.0 + 0.5 + blockIdx.x * blockDim.x +
                         blockDim.x / 2.0) *
                        dx;
    const double ymid = (image_height / 2.0 - 0.5 - blockIdx.y * blockDim.y -
                         blockDim.y / 2.0) *
                        dy;
    const float3 central_pulse_pos = ant_pos[num_pulses / 2];
    const double cxdiff = (double)central_pulse_pos.x - xmid;
    const double cydiff = (double)central_pulse_pos.y - ymid;
    const double czdiff = (double)central_pulse_pos.z - z0;
    // Precompute the distance from the central pulse to the central pixel to
    // use as an initialization distance for Newton-Raphson updates.
    const double r_mid_sqr =
        cxdiff * cxdiff + cydiff * cydiff + czdiff * czdiff;
    const double Rmid = sqrt(r_mid_sqr);
    const double Rmid_2_inv = 1.0 / (2.0 * Rmid);

    const int t = threadIdx.y * blockDim.x + threadIdx.x;
    const int nthreads = blockDim.x * blockDim.y;
    for (int p = t; p < num_pulses; p += nthreads) {
        const float3 ap = ant_pos[p];
        const double apx = ap.x;
        const double apy = ap.y;
        const double apz = ap.z;
        smem_dists[2 * p] = sqrt(apx * apx + apy * apy + apz * apz);
        smem_dists[2 * p + 1] =
            apx * apx + apy * apy + apz * apz - 2.0 * xmid * apx + xmid * xmid -
            2.0 * ymid * apy + ymid * ymid - 2.0 * apz * z0 + z0 * z0;
    }

    block.sync();

    if (ix >= image_width || iy >= image_height) {
        return;
    }

    const float xdel = (-image_width / 2.0f + 0.5f + ix) * dx - xmid;
    const float ydel = (image_height / 2.0f - 0.5f - iy) * dy - ymid;
    const double dist_adj =
        2.0 * xmid * xdel + xdel * xdel + 2.0 * ymid * ydel + ydel * ydel;

    cuComplex accum = make_cuComplex(0.0, 0.0);
    const float bin_offset = 0.5f * num_range_bins;
    const float max_bin_f = num_range_bins - 2.0f;
    for (int p = 0; p < num_pulses; ++p) {
        const float3 ap = ant_pos[p];
        const double range_to_center = smem_dists[2 * p];
        const double dx2dy2dz2 =
            smem_dists[2 * p + 1] + dist_adj -
            (double)(2.0f * ap.x * xdel + 2.0f * ap.y * ydel);

        // Newton-Raphson updates to compute distance to the pixel. The second
        // iteration is approximate because it reuses the denominator for the
        // first iterative to avoid the double precision division to
        // compute 1.0/(2.0*nr_1).
        // const double dx2dy2dz2 = xdiff*xdiff+ydiff*ydiff+zdiff*zdiff;
        const double nr_1 = Rmid - (Rmid * Rmid - dx2dy2dz2) * Rmid_2_inv;
        const double nr_2 = nr_1 - (nr_1 * nr_1 - dx2dy2dz2) * Rmid_2_inv;
        const double diffR = nr_2 - range_to_center;

        const int pulse_offset = p * num_range_bins;

        const float bin = (float)(diffR * dr_inv) + bin_offset;
        if (bin >= 0.0f && bin < max_bin_f) {
            const int bin_floor = (int)bin;
            const float w = (bin - (float)bin_floor);
            const int ind_base = pulse_offset + bin_floor;
            const cuComplex sample =
                make_cuComplex((1.0f - w) * range_profiles[ind_base].x +
                                   w * range_profiles[ind_base + 1].x,
                               (1.0f - w) * range_profiles[ind_base].y +
                                   w * range_profiles[ind_base + 1].y);

            double sinx, cosx;
            sincos(phase_correction_partial * diffR, &sinx, &cosx);
            const cuComplex matched_filter = make_cuComplex(cosx, sinx);
            accum = cuCfmaf(sample, matched_filter, accum);
        }
    }

    const int img_ind = iy * image_width + ix;
    image[img_ind] = cuCaddf(image[img_ind], accum);
}

__global__ void
SarBpKernelIncrPhaseLUT(cuComplex *image, int image_width, int image_height,
                        const cuComplex *range_profiles,
                        const cuComplex *incr_phase_lut, int num_range_bins,
                        int num_pulses, const float3 *ant_pos,
                        double phase_correction_partial, double dr_inv,
                        double dx, double dy, double z0) {
    extern __shared__ double smem_dists[];

    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    cg::thread_block block = cg::this_thread_block();

    const double xmid = (-image_width / 2.0 + 0.5 + blockIdx.x * blockDim.x +
                         blockDim.x / 2.0) *
                        dx;
    const double ymid = (image_height / 2.0 - 0.5 - blockIdx.y * blockDim.y -
                         blockDim.y / 2.0) *
                        dy;
    const float3 central_pulse_pos = ant_pos[num_pulses / 2];
    const double cxdiff = (double)central_pulse_pos.x - xmid;
    const double cydiff = (double)central_pulse_pos.y - ymid;
    const double czdiff = (double)central_pulse_pos.z - z0;
    // Precompute the distance from the central pulse to the central pixel to
    // use as an initialization distance for Newton-Raphson updates.
    const double r_mid_sqr =
        cxdiff * cxdiff + cydiff * cydiff + czdiff * czdiff;
    const double Rmid = sqrt(r_mid_sqr);
    const double Rmid_2_inv = 1.0 / (2.0 * Rmid);

    phase_correction_partial *= 1.0 / dr_inv;

    const int t = threadIdx.y * blockDim.x + threadIdx.x;
    const int nthreads = blockDim.x * blockDim.y;
    for (int p = t; p < num_pulses; p += nthreads) {
        const float3 ap = ant_pos[p];
        const double apx = ap.x;
        const double apy = ap.y;
        const double apz = ap.z;
        smem_dists[2 * p] = sqrt(apx * apx + apy * apy + apz * apz);
        smem_dists[2 * p + 1] =
            apx * apx + apy * apy + apz * apz - 2.0 * xmid * apx + xmid * xmid -
            2.0 * ymid * apy + ymid * ymid - 2.0 * apz * z0 + z0 * z0;
    }

    block.sync();

    if (ix >= image_width || iy >= image_height) {
        return;
    }

    const float xdel = (-image_width / 2.0 + 0.5 + ix) * dx - xmid;
    const float ydel = (image_height / 2.0 - 0.5 - iy) * dy - ymid;
    const double dist_adj =
        2.0 * xmid * xdel + xdel * xdel + 2.0 * ymid * ydel + ydel * ydel;

    cuComplex accum = make_cuComplex(0.0, 0.0);
    const double bin_offset = (double)(0.5f * num_range_bins);
    const float max_bin_f = num_range_bins - 2.0f;
    for (int p = 0; p < num_pulses; ++p) {
        const float3 ap = ant_pos[p];
        const double range_to_center = smem_dists[2 * p];
        const double dx2dy2dz2 =
            smem_dists[2 * p + 1] + dist_adj -
            (double)(2.0f * ap.x * xdel + 2.0f * ap.y * ydel);

        // Newton-Raphson updates to compute distance to the pixel. The second
        // iteration is approximate because it reuses the denominator for the
        // first iterative to avoid the double precision division to
        // compute 1.0/(2.0*nr_1).
        // const double dx2dy2dz2 = xdiff*xdiff+ydiff*ydiff+zdiff*zdiff;
        const double nr_1 = Rmid - (Rmid * Rmid - dx2dy2dz2) * Rmid_2_inv;
        const double nr_2 = nr_1 - (nr_1 * nr_1 - dx2dy2dz2) * Rmid_2_inv;
        const double diffR = nr_2 - range_to_center;

        const int pulse_offset = p * num_range_bins;

        const double bin = (diffR * dr_inv) + bin_offset;
        const float bin_f = (float)bin;
        if (bin_f >= 0.0f && bin_f < max_bin_f) {
            const int bin_floor = (int)bin;
            const float w = (float)(bin - (double)bin_floor);
            const int ind_base = pulse_offset + bin_floor;
            const cuComplex sample =
                make_cuComplex((1.0f - w) * range_profiles[ind_base].x +
                                   w * range_profiles[ind_base + 1].x,
                               (1.0f - w) * range_profiles[ind_base].y +
                                   w * range_profiles[ind_base + 1].y);
            float sinx, cosx;
            const cuComplex first_part = incr_phase_lut[bin_floor];
            __sincosf(phase_correction_partial * w, &sinx, &cosx);
            const cuComplex second_part = make_cuComplex(cosx, sinx);
            accum = cuCfmaf(sample, cuCmulf(first_part, second_part), accum);
        }
    }

    const int img_ind = iy * image_width + ix;
    image[img_ind] = cuCaddf(image[img_ind], accum);
}

__global__ void
SarBpKernelSinglePrecision(cuComplex *image, int image_width, int image_height,
                           const cuComplex *range_profiles, int num_range_bins,
                           int num_pulses, const float3 *ant_pos,
                           float phase_correction_partial, float dr_inv,
                           float dx, float dy, float z0) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= image_width || iy >= image_height) {
        return;
    }

    const float py = (image_height / 2.0f - 0.5f - iy) * dy;
    const float px = (-image_width / 2.0f + 0.5f + ix) * dx;

    cuComplex accum = make_cuComplex(0.0f, 0.0f);
    const float bin_offset = 0.5f * num_range_bins;
    const float max_bin_f = num_range_bins - 2.0f;
    for (int p = 0; p < num_pulses; ++p) {
        const int pulse_offset = p * num_range_bins;
        const float3 ap = ant_pos[p];
        const float apx = ap.x;
        const float apy = ap.y;
        const float apz = ap.z;
        const float range_to_center = sqrtf(apx * apx + apy * apy + apz * apz);
        const float xdiff = apx - px;
        const float ydiff = apy - py;
        const float zdiff = apz - z0;
        const float diffR =
            sqrtf(xdiff * xdiff + ydiff * ydiff + zdiff * zdiff) -
            range_to_center;

        const float bin = diffR * dr_inv + bin_offset;
        if (bin >= 0.0f && bin < max_bin_f) {
            const int bin_floor = (int)bin;
            const float w = (bin - (float)bin_floor);
            const int ind_base = pulse_offset + bin_floor;
            const cuComplex sample =
                make_cuComplex((1.0f - w) * range_profiles[ind_base].x +
                                   w * range_profiles[ind_base + 1].x,
                               (1.0f - w) * range_profiles[ind_base].y +
                                   w * range_profiles[ind_base + 1].y);

            float sinx, cosx;
            __sincosf(phase_correction_partial * diffR, &sinx, &cosx);
            const cuComplex matched_filter = make_cuComplex(cosx, sinx);
            accum = cuCfmaf(sample, matched_filter, accum);
        }
    }

    const int img_ind = iy * image_width + ix;
    image[img_ind].x += accum.x;
    image[img_ind].y += accum.y;
}

__global__ void FindMaxMagnitudePhaseOneKernel(float *max_magnitude_workbuf,
                                               const cuComplex *src_image,
                                               int image_width,
                                               int image_height) {
    extern __shared__ float smem[];
    const int ix = threadIdx.x;
    const int nx = blockDim.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile tile = cg::tiled_partition<32>(block);

    float max = 0.0f;
    if (row < image_height) {
        for (int x = ix; x < image_width; x += nx) {
            const cuComplex val = src_image[row * image_width + x];
            const float mag = sqrtf(val.x * val.x + val.y * val.y);
            if (mag > max) {
                max = mag;
            }
        }
    }

    max = cg::reduce(tile, max, cg::greater<float>());
    if (tile.thread_rank() == 0) {
        smem[tile.meta_group_rank()] = max;
    }

    block.sync();

    if (block.thread_rank() == 0) {
        for (auto i = 0; i < tile.meta_group_size(); i++) {
            if (smem[i] > max) {
                max = smem[i];
            }
        }
        max_magnitude_workbuf[blockIdx.y] = max;
    }
}

__global__ void FindMaxMagnitudePhaseTwoKernel(float *max_magnitude_workbuf,
                                               int num_phase_one_vals) {
    const int ix = threadIdx.x;
    const int nx = blockDim.x;

    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile tile = cg::tiled_partition<32>(block);

    float max = 0.0f;
    for (int i = ix; i < num_phase_one_vals; i += nx) {
        if (max_magnitude_workbuf[i] > max) {
            max = max_magnitude_workbuf[i];
        }
    }

    max = cg::reduce(tile, max, cg::greater<float>());
    if (tile.thread_rank() == 0) {
        max_magnitude_workbuf[0] = max;
    }
}

__global__ void ComputeMagnitudeImageKernel(uint32_t *dest_image,
                                            const cuComplex *src_image,
                                            int image_width, int image_height,
                                            const float *max_magnitude,
                                            float min_normalized_db) {
    const int ix = threadIdx.x;
    const int iy = threadIdx.y;
    const int nx = blockDim.x;
    const int ny = blockDim.y;
    const float max_mag = *max_magnitude;

    const float normalized_db_inv = -1.0f / min_normalized_db;
    for (int y = iy; y < image_height; y += ny) {
        for (int x = ix; x < image_width; x += nx) {
            const int pix_ind = y * image_width + x;
            const cuComplex val = src_image[pix_ind];
            const float mag = sqrtf(val.x * val.x + val.y * val.y);
            const float db = 20.0f * log10f(mag / max_mag);
            const float db_normalized = db * normalized_db_inv;
            const uint v =
                static_cast<uint>(__saturatef(db_normalized + 1.0f) * 255.0f) &
                0xff;
            dest_image[pix_ind] = (0xFFu << 24) | (v << 16) | (v << 8) | v;
        }
    }
}

__global__ void ResampleMagnitudeImageKernel(uint32_t *resampled,
                                             const uint32_t *src_image,
                                             int resample_width,
                                             int resample_height, int src_width,
                                             int src_height) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= resample_width || iy >= resample_height) {
        return;
    }

    const float x = static_cast<float>(ix) * src_width / resample_width;
    const float y = static_cast<float>(iy) * src_height / resample_height;

    const float xf = floorf(x);
    const float yf = floorf(y);

    const float x_alpha = x - xf;
    const float y_alpha = y - yf;

    const int sx = (int)xf;
    const int sy = (int)yf;

    const float w1 = (1.0f - x_alpha) * (1.0f - y_alpha);
    const float w2 = x_alpha * (1.0f - y_alpha);
    const float w3 = (1.0f - x_alpha) * y_alpha;
    const float w4 = x_alpha * y_alpha;

    union rgba {
        uint32_t u32;
        uint8_t u8[4];
    };
    rgba p1, p2, p3, p4;

    const int base_ind = sy * src_width + sx;
    p1.u32 = src_image[base_ind];
    p2.u32 = (sx < src_width - 1) ? src_image[base_ind + 1] : 0;
    p3.u32 = (sy < src_height - 1) ? src_image[base_ind + src_width] : 0;
    p4.u32 = (sx < src_width - 1 && sy < src_height - 1)
                 ? src_image[base_ind + src_width + 1]
                 : 0;
    rgba accum = {.u32 = 0};
    for (int k = 0; k < 3; k++) {
        const float val =
            w1 * p1.u8[k] + w2 * p2.u8[k] + w3 * p3.u8[k] + w4 * p4.u8[k];
        accum.u8[k] = (val < 0.0f) ? 0 : (val >= 255.0f) ? 255 : roundf(val);
    }
    accum.u8[3] = 255;
    resampled[iy * resample_width + ix] = accum.u32;
}

void FftShiftGpu(cuComplex *data, int num_samples, int num_arrays,
                 cudaStream_t stream) {
    const dim3 block(128, 1);
    const dim3 grid(num_arrays, 1, 1);
    FftShiftKernel<<<grid, block, 0, stream>>>(data, num_samples, num_arrays);
    cudaChecked(cudaGetLastError());
}

static int SelectedKernelPulseBlockSize(SarGpuKernel kernel, int num_pulses) {
    if (kernel == SarGpuKernel::SmemRange ||
        kernel == SarGpuKernel::IncrPhaseLookup) {
        const size_t maxSmemPerBlock = 8 * 1024;
        const int max_pulses = (maxSmemPerBlock / sizeof(double) / 2);
        return (max_pulses >= num_pulses) ? num_pulses : max_pulses;
    }
    return num_pulses;
}

size_t GetMaxBackprojWorkBufSizeBytes(int num_range_bins) {
    const size_t incr_phase_lookup_size =
        sizeof(cuDoubleComplex) * num_range_bins;
    // Currently, only the IncrPhaseLookup kernel requires a work buffer, but
    // if we add additional kernels that require work buffers, we should return
    // the max of them here.
    return incr_phase_lookup_size;
}

__global__ void SarBpPopulateIncrPhaseLUT(cuComplex *workbuf,
                                          int num_range_bins, double freq_min,
                                          double dr) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num_range_bins) {
        return;
    }

    const double c = SPEED_OF_LIGHT_M_PER_SEC;
    // The data has been motion-compenstated so that the origin corresponds
    // to DC. The phase compenstation for a pixel with distance delR from
    // the origin is thus:
    //   exp(1i * 4 * pi * minum_freq * delR / c)
    // Note that delR = 0 yields a value of 1 (no phase correction).
    // Here, we precompute a LUT of
    //   exp(1i * 4 * pi * minum_freq * R_b / c)
    // for each range bin start distance R_b and where delR = R_b + w * dR,
    // w is the interpolation weight, and dR is the distance from range bin
    // to range bin.
    // Ultimately, we want to compute
    //   exp(1i * 4 * pi * minum_freq * (R_b + w * dR) / c)
    // which we decompose into
    // exp(1i * 4 * pi * minum_freq * R_b / c) + exp(1i * 4 * pi * minum_freq *
    // w * dR / c) The first values are precomputed here at double precision and
    // the second values, for which R_frac is small, will be computed using
    // single precision and the trigonometric intrinsics. The argument of the
    // second expression is precomputed in the kernel, except for the w, which
    // varies per backprojection.
    const double r_bin = (t - 0.5 * num_range_bins) * dr;
    const double phase_correction_partial = 4.0 * M_PI * (freq_min / c) * r_bin;
    double sinx, cosx;
    sincos(phase_correction_partial, &sinx, &cosx);
    workbuf[t] = make_cuComplex(cosx, sinx);
}

static bool s_did_init_phase_lut = false;
static void PopulateBpWorkbuf(uint8_t *bp_workbuf, int num_range_bins,
                              double freq_min, double dr, SarGpuKernel kernel,
                              cudaStream_t stream) {
    if (s_did_init_phase_lut || kernel != SarGpuKernel::IncrPhaseLookup) {
        return;
    }

    const dim3 block(128, 1);
    const dim3 grid(idivup(num_range_bins, block.x), 1);
    SarBpPopulateIncrPhaseLUT<<<grid, block, 0, stream>>>(
        reinterpret_cast<cuComplex *>(bp_workbuf), num_range_bins, freq_min,
        dr);

    s_did_init_phase_lut = true;
}

void SarBpGpu(cuComplex *image, int image_width, int image_height,
              const cuComplex *range_profiles, uint8_t *bp_workbuf,
              int num_range_bins, int num_pulses, const float3 *ant_pos,
              double freq_min, double dR, double dx, double dy, double z0,
              SarGpuKernel kernel, cudaStream_t stream) {
    const dim3 block(16, 16);
    const dim3 grid(idivup(image_width, block.x), idivup(image_height, block.y),
                    1);

    PopulateBpWorkbuf(bp_workbuf, num_range_bins, freq_min, dR, kernel, stream);
    cudaChecked(cudaGetLastError());

    const double c = SPEED_OF_LIGHT_M_PER_SEC;
    // The data has been motion-compenstated so that the origin corresponds
    // to DC. The phase compenstation for a pixel with distance delR from
    // the origin is thus:
    //   exp(1i * 4 * pi * minum_freq * delR / c)
    // Note that delR = 0 yields a value of 1 (no phase correction).
    // Everything except delR is fixed, so we precompute it.
    const double phase_correction_partial = 4.0 * M_PI * (freq_min / c);

    const int pulse_block_size =
        SelectedKernelPulseBlockSize(kernel, num_pulses);
    int num_processed_pulses = 0;
    while (num_processed_pulses < num_pulses) {
        const int num_pulses_this_block =
            std::min(num_pulses - num_processed_pulses, pulse_block_size);
        switch (kernel) {
        case SarGpuKernel::DoublePrecision:
            SarBpKernelDoublePrecision<<<grid, block, 0, stream>>>(
                image, image_width, image_height,
                range_profiles + num_processed_pulses * num_range_bins,
                num_range_bins, num_pulses_this_block,
                ant_pos + num_processed_pulses, phase_correction_partial,
                1.0 / dR, dx, dy, z0);
            break;
        case SarGpuKernel::MixedPrecision:
            SarBpKernelMixedPrecision<<<grid, block, 0, stream>>>(
                image, image_width, image_height,
                range_profiles + num_processed_pulses * num_range_bins,
                num_range_bins, num_pulses_this_block,
                ant_pos + num_processed_pulses, phase_correction_partial,
                1.0 / dR, dx, dy, z0);
            break;
        case SarGpuKernel::SmemRange: {
            const size_t smemSize = sizeof(double) * 2 * num_pulses_this_block;
            SarBpKernelSmemRange<<<grid, block, smemSize, stream>>>(
                image, image_width, image_height,
                range_profiles + num_processed_pulses * num_range_bins,
                num_range_bins, num_pulses_this_block,
                ant_pos + num_processed_pulses, phase_correction_partial,
                1.0 / dR, dx, dy, z0);
        } break;
        case SarGpuKernel::IncrPhaseLookup: {
            const size_t smemSize = sizeof(double) * 2 * num_pulses_this_block;
            SarBpKernelIncrPhaseLUT<<<grid, block, smemSize, stream>>>(
                image, image_width, image_height,
                range_profiles + num_processed_pulses * num_range_bins,
                reinterpret_cast<cuComplex *>(bp_workbuf), num_range_bins,
                num_pulses_this_block, ant_pos + num_processed_pulses,
                phase_correction_partial, 1.0 / dR, dx, dy, z0);
        } break;
        case SarGpuKernel::SinglePrecision:
            SarBpKernelSinglePrecision<<<grid, block, 0, stream>>>(
                image, image_width, image_height,
                range_profiles + num_processed_pulses * num_range_bins,
                num_range_bins, num_pulses_this_block,
                ant_pos + num_processed_pulses, phase_correction_partial,
                1.0 / dR, dx, dy, z0);
            break;
        case SarGpuKernel::Invalid:
            LOG_ERR("Invalid SAR BP kernel selection");
            break;
        }
        cudaChecked(cudaGetLastError());
        num_processed_pulses += num_pulses_this_block;
    }
}

size_t GetMaxMagnitudeWorkBufSizeBytes(int image_height) {
    return sizeof(float) * idivup(image_height, 16);
}

void ComputeMagnitudeImage(uint *dest_image, float *max_magnitude_workbuf,
                           const cuComplex *src_image, int image_width,
                           int image_height, float min_normalized_db,
                           cudaStream_t stream) {
    const dim3 block(16, 16);
    const dim3 grid(1, idivup(image_height, block.y));

    const size_t smem_size = sizeof(float) * idivup(block.x * block.y, 32);
    FindMaxMagnitudePhaseOneKernel<<<grid, block, smem_size, stream>>>(
        max_magnitude_workbuf, src_image, image_width, image_height);

    FindMaxMagnitudePhaseTwoKernel<<<dim3(1, 1), dim3(32, 1), 0, stream>>>(
        max_magnitude_workbuf, idivup(image_height, block.y));

    ComputeMagnitudeImageKernel<<<grid, block, 0, stream>>>(
        dest_image, src_image, image_width, image_height, max_magnitude_workbuf,
        min_normalized_db);
}

void ResampleMagnitudeImage(uint32_t *resampled, const uint32_t *src_image,
                            int resample_width, int resample_height,
                            int src_width, int src_height,
                            cudaStream_t stream) {
    const dim3 block(16, 16);
    const dim3 grid(idivup(resample_width, block.x),
                    idivup(resample_height, block.y));
    ResampleMagnitudeImageKernel<<<grid, block, 0, stream>>>(
        resampled, src_image, resample_width, resample_height, src_width,
        src_height);
    cudaChecked(cudaGetLastError());
}