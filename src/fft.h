#ifndef _FFT_H_
#define _FFT_H_

#include <cstdint>
#include <vector>

#include <cuComplex.h>

#include "common.h"

class FftRadix2 {
  public:
    // Apply the transform to data. If inverse is true, then apply
    // the inverse transform. This is a radix-2 implementation
    // (a fairly straightforward Cooley-Tukey implementation) and
    // only supports powers of two. It is used as a reference implementation
    // and is not optimized.
    // Returns SarReturnCode::Success (0) on success and
    // SarReturnCode::InvalidArg if the data length is not
    // a power of two.
    SarReturnCode Transform(std::vector<cuComplex> &data, bool inverse);

  private:
    // Lazily initialize the pre-computed twiddle factors. This
    // implementation is not intended for cases where the input
    // size changes very frequently.
    // Returns SarReturnCode::Success (0) on success and
    // SarReturnCode::InvalidArg if the data length is not
    // a power of two.
    SarReturnCode Init(size_t transform_size);

    uint32_t reverseBits(uint32_t x, uint32_t order);

    uint32_t m_transform_size{0};
    uint32_t m_order{0};
    std::vector<cuComplex> m_exp_table;
};

#endif // _FFT_H_