#include <algorithm>
#include <cassert>
#include <cmath>

#include "ser.h"

double signal_to_error_ratio(const std::vector<cuComplex> &reference,
                             const std::vector<cuComplex> &test) {
    assert(reference.size() == test.size());
    double sum_square_diff = 0.0, sum_square_ref = 0.0;
    const size_t N = std::min(reference.size(), test.size());
    for (size_t i = 0; i < N; i++) {
        const double re_diff = reference[i].x - test[i].x;
        const double im_diff = reference[i].y - test[i].y;
        sum_square_diff += re_diff * re_diff + im_diff * im_diff;
        sum_square_ref +=
            reference[i].x * reference[i].x + reference[i].y * reference[i].y;
    }
    if (sum_square_diff == 0.0 || sum_square_ref == 0.0) {
        return -140.0;
    }
    return 20 * log10(sqrt(sum_square_ref / sum_square_diff));
}