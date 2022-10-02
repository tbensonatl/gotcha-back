#include "fft.h"
#include "helpers.h"

#include <cmath>

static inline uint32_t getOrderOfPowerOf2(uint32_t x) {
    uint32_t order = 0;
    while (x > 1) {
        x /= 2;
        order++;
    }
    return order;
}

uint32_t FftRadix2::reverseBits(uint32_t x, uint32_t order) {
    uint32_t result = 0;
    for (uint32_t i = 0; i < order; i++, x >>= 1) {
        result = (result << 1) | (x & 1);
    }
    return result;
}

SarReturnCode FftRadix2::Init(size_t transform_size) {
    // We do not support FFTs of size larger than 2^32-1, or
    // really 2^31 since we later verify that the transform sizes
    // are powers of two.
    if (transform_size > std::numeric_limits<uint32_t>::max()) {
        return SarReturnCode::InvalidArg;
    }
    const uint32_t n = transform_size;
    if (n == 0 || ((n & (n - 1)) != 0)) {
        // We only support powers-of-two. More generically, we could support
        // Bluestein transforms to extend support, but in practice we use
        // only power-of-two transforms for the range upsampling.
        m_transform_size = 0;
        m_exp_table.clear();
        return SarReturnCode::InvalidArg;
    }

    m_transform_size = n;
    m_order = getOrderOfPowerOf2(m_transform_size);
    m_exp_table.resize(m_transform_size / 2);
    for (uint32_t i = 0; i < m_transform_size / 2; i++) {
        const double arg = -2.0 * M_PI * i / m_transform_size;
        double sinx, cosx;
        sincos(arg, &sinx, &cosx);
        // Pre-compute the twiddle factors (i.e. the complex roots of unity)
        m_exp_table[i].x = cosx;
        m_exp_table[i].y = sinx;
    }
    return SarReturnCode::Success;
}

SarReturnCode FftRadix2::Transform(std::vector<cuComplex> &data, bool inverse) {
    if (static_cast<size_t>(m_transform_size) != data.size()) {
        const SarReturnCode rc = Init(data.size());
        if (rc != SarReturnCode::Success) {
            return rc;
        }
    }

    // Bit reversal permutation
    for (uint32_t i = 0; i < m_transform_size; i++) {
        const uint32_t j = reverseBits(i, m_order);
        if (j > i) {
            std::swap(data[i], data[j]);
        }
    }
    const uint32_t n = m_transform_size;

    // Cooley-Tukey decimation-in-time radix-2 FFT. This loop follows the
    // pseudo-code
    // shown (as of Sep 30 2022) at:
    //   https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm#Data_reordering,_bit_reversal,_and_in-place_algorithms
    // For clarity, we use the same loop variables as the above reference, but
    // the omega complex exponential values that the pseudocode generates
    // incrementally are instead read from the pre-computed table in this
    // implementation.
    for (size_t m = 2; m <= n; m *= 2) {
        const size_t half_m = m / 2;
        // We use the set of complex exponentials exp(-2*pi*i/m), so
        // we need to stride through the pre-computed table in steps
        // of n/m.
        const size_t exp_table_step = n / m;
        for (size_t k = 0; k < n; k += m) {
            for (size_t j = 0, p = 0; j < m / 2; j++, p += exp_table_step) {
                // The only difference between the complex exponentials used for
                // forward and inverse transforms is that the latter are
                // conjugated relative to the pre-computed values.
                const cuComplex twiddle_factor =
                    (inverse) ? cuConjf(m_exp_table[p]) : m_exp_table[p];
                const cuComplex tmp =
                    cuCmulf(data[k + j + half_m], twiddle_factor);
                data[k + j + half_m] = cuCsubf(data[k + j], tmp);
                data[k + j] = cuCaddf(data[k + j], tmp);
            }
        }
        if (m >= n) {
            // Avoid potential overflow on last m *= 2 calc in the for loop
            break;
        }
    }

    return SarReturnCode::Success;
}