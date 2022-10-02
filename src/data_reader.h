#ifndef _DATA_READER_H_
#define _DATA_READER_H_

#include <string>
#include <vector>

#include <cuComplex.h>
#include <vector_types.h>

#include "common.h"

/*
 * The following document describes the MATLAB file format:
 *
 *  https://www.mathworks.com/help/pdf_doc/matlab/matfile_format.pdf
 *
 * The GOTCHA data is in MATLAB version 5.0. The MATLAB file support included
 * in this project is the minimal set needed to extract the required data from
 * the GOTCHA .mat data files. For more complete MATLAB file support, search
 * for the matio project on GitHub.
 */

#include <string>

namespace Gotcha {

enum class Polarization { HH, HV, VH, VV };

std::string PolarizationToString(Polarization p);

struct DataRequest {
    int first_az{0};
    int last_az{0};
    int pass{1};
    Polarization polarization{Polarization::HH};
};

struct DataBlock {
    std::vector<cuComplex> phase_history;
    std::vector<float3> ant_pos;
    struct DataRequest data_request;
    int num_pulses{0};
    int num_freq{0};
    float del_freq{0.0f};
    float min_freq{0.0f};
    float del_r{0.0f};

    void Reset() {
        phase_history.clear();
        ant_pos.clear();
    }
};

class Reader {
  public:
    Reader(const std::string &gotcha_dir) : m_gotcha_dir(gotcha_dir) {}
    SarReturnCode ReadDataSet(struct DataBlock &block,
                              const struct DataRequest &request);

  private:
    std::string m_gotcha_dir;
};

} // namespace Gotcha

#endif /* _DATA_READER_H_ */