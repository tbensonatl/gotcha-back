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

// Data stored in a Single File Format (SFF). This is a simple file format
// primarily intended to ease testing.
namespace SFF {
// SFF consists of data in the following order and format:
// (1) The header below,
// (2) num_pulses pulses, each with num_freq samples, using two FP32 values
// (I/Q) per sample
// (3) num_pulses antenna positions, each with 3 coordinates
// (x,y,z) in FP32 format All of the fields are in little endian format.
struct Header {
    uint32_t num_pulses;
    uint32_t num_freq;
    float del_freq;
    float min_freq;
    uint32_t flags;
    uint32_t pad[3]; // pad header to 32 bytes
};

enum class Flags {
    // Set if the output file has data for the pulses. A file that only includes
    // a header and antenna positions will have the data populated via a seeded
    // random number generator on read. This provides a useful way for
    // performance
    // testing without requiring large files.
    FileHasData = 0x1
};

// Write a dataset represented as a Gotcha::DataBlock into an SFF output file.
// If skip_data is true, then only the header and antenna positions will be
// written and the header will have the SFF::Flags::FileHasData flag set to
// false.
void WriteDatasetAsSFF(const char *filename, const Gotcha::DataBlock &data,
                       bool skip_data = false);
void ReadDataset(Gotcha::DataBlock &data, const char *filename);
} // namespace SFF

#endif /* _DATA_READER_H_ */