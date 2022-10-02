#include <vector>

#include <cuComplex.h>

double signal_to_error_ratio(const std::vector<cuComplex> &reference,
                             const std::vector<cuComplex> &test);