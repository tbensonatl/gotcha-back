#include "data_reader.h"
#include "helpers.h"

#include <filesystem>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

namespace fs = std::filesystem;

enum DataType {
    miINT8 = 1,
    miUINT8 = 2,
    miINT16 = 3,
    miUINT16 = 4,
    miINT32 = 5,
    miUINT32 = 6,
    miSINGLE = 7,
    miDOUBLE = 9,
    miINT64 = 12,
    miUINT64 = 13,
    miMATRIX = 14,
    miCOMPRESSED = 15,
    miUTF8 = 16,
    miUTF16 = 17,
    miUTF32 = 18
};

enum ClassType {
    mxCELL = 1,
    mxSTRUCT = 2,
    mxOBJECT = 3,
    mxCHAR = 4,
    mxSPARSE = 5,
    mxDOUBLE = 6,
    mxSINGLE = 7,
    mxINT8 = 8,
    mxUINT8 = 9,
    mxINT16 = 10,
    mxUINT16 = 11,
    mxINT32 = 12,
    mxUINT32 = 13,
    mxINT64 = 14,
    mxUINT64 = 15
};

struct DataHeader {
    // Text in human readable format
    unsigned char human[116];
    // Unusued in the GOTCHA data
    uint8_t subsystem_data_offest[8];
    // 0x100 in the GOTCHA data
    uint16_t version;
    // Characters M and I written to the file in-order. If read as IM when
    // reading as a 16-bit value, then byte-swapping is required.
    uint16_t endian_indicator;
};

struct DataElementTag {
    uint32_t type;
    uint32_t num_bytes;
};

static ClassType GetClassTypeFromArrayFlags(uint32_t flags) {
    switch ((flags & 0xFF)) {
    case mxCELL:
        return mxCELL;
    case mxSTRUCT:
        return mxSTRUCT;
    case mxOBJECT:
        return mxOBJECT;
    case mxCHAR:
        return mxCHAR;
    case mxSPARSE:
        return mxSPARSE;
    case mxDOUBLE:
        return mxDOUBLE;
    case mxSINGLE:
        return mxSINGLE;
    case mxINT8:
        return mxINT8;
    case mxUINT8:
        return mxUINT8;
    case mxINT16:
        return mxINT16;
    case mxUINT16:
        return mxUINT16;
    case mxINT32:
        return mxINT32;
    case mxUINT32:
        return mxUINT32;
    case mxINT64:
        return mxINT64;
    case mxUINT64:
        return mxUINT64;
    default:
        throw std::runtime_error("invalid class type " + std::to_string(flags));
    }
}

static size_t ReadDataElementTag(FILE *fp, DataElementTag &tag) {
    const size_t n = fread(&tag, sizeof(DataElementTag), 1, fp);
    if (n != 1) {
        throw std::runtime_error("invalid read for data element tag");
    }
    return sizeof(DataElementTag);
}

static size_t ReadDataElement(FILE *fp, std::vector<uint8_t> &data,
                              DataElementTag &tag) {
    size_t nread = 0;

    nread += ReadDataElementTag(fp, tag);

    // Check for a small data element format
    if ((tag.type & 0xFFFF0000) != 0) {
        const uint32_t type = tag.type & 0xFFFF;
        const uint32_t num_bytes = (tag.type & 0xFFFF0000) >> 16;
        data.resize(num_bytes);
        for (uint32_t i = 0; i < num_bytes; i++) {
            const uint32_t shift = i * 8;
            data[i] = (tag.num_bytes & (0xFF << shift)) >> shift;
        }
        tag.type = type;
        tag.num_bytes = num_bytes;
    } else {
        data.resize(tag.num_bytes);
        if (fread(&data[0], sizeof(uint8_t), tag.num_bytes, fp) !=
            tag.num_bytes) {
            throw std::runtime_error("invalid read for element data");
        }
        nread += tag.num_bytes;
    }

    return nread;
}

// MAT 5.0 file data elements are aligned to 8-byte boundaries, so skip
// any padding required between data elements so that the next data
// element is aligned to an 8-byte boundary
static size_t SkipPadding(FILE *fp, size_t nread) {
#define MAT_FILE_ALIGNMENT (8)
    uint8_t pad[MAT_FILE_ALIGNMENT];
    size_t pad_size = 0;
    if (nread % MAT_FILE_ALIGNMENT > 0) {
        pad_size = MAT_FILE_ALIGNMENT - nread % MAT_FILE_ALIGNMENT;
        if (fread(pad, sizeof(uint8_t), pad_size, fp) != pad_size) {
            throw std::runtime_error("read error for padding");
        }
    }
    return pad_size;
}

// We only handle the Gotcha data, so there are many limitations and error
// checks while reading the files. Furthermore, we do not support nested
// structs, even though the Gotcha data uses these for the autofocus data. We do
// not use the autofocus values, so we simply skip those elements.
static void ReadMatrixData(FILE *fp, Gotcha::DataBlock &block,
                           int64_t remaining_bytes) {
    DataElementTag tag;
    // We use real_data for real numeric data, imag_data for imaginary
    // numeric data (in the case of complex arrays), and work_buf for
    // all other data reads.
    std::vector<uint8_t> work_buf, real_data, imag_data;

    ssize_t nread = 0;

    // Start of the first data element, which is a matrix with class type
    // mxSTRUCT
    nread += ReadDataElementTag(fp, tag);
    if (tag.type != miMATRIX) {
        throw std::runtime_error("expected miMATRIX container type, got " +
                                 std::to_string(tag.type));
    }

    // Array flags. Should be an 8 byte miUINT32 data element.
    nread += ReadDataElement(fp, work_buf, tag);
    if (tag.type != miUINT32) {
        throw std::runtime_error("sub tag type is " + std::to_string(tag.type) +
                                 "; expected miUINT32 for flags");
    }
    if (tag.num_bytes != 8) {
        throw std::runtime_error("expected 8 bytes for flags, got " +
                                 std::to_string(tag.num_bytes));
    }

    const uint32_t *const array_flags =
        reinterpret_cast<uint32_t *>(&work_buf[0]);
    const ClassType class_type = GetClassTypeFromArrayFlags(array_flags[0]);
    if (class_type != mxSTRUCT) {
        throw std::runtime_error(
            "expected struct for outermost data element, got " +
            std::to_string(class_type));
    }

    // Data dimensions
    nread += ReadDataElement(fp, work_buf, tag);
    const int num_dimensions = work_buf.size() / sizeof(int32_t);
    std::vector<int32_t> dimensions(num_dimensions);
    const int32_t *const ptr = reinterpret_cast<int32_t *>(&work_buf[0]);
    for (int i = 0; i < num_dimensions; i++) {
        dimensions[i] = ptr[i];
    }

    // Array name
    nread += ReadDataElement(fp, work_buf, tag);
    std::string name(work_buf.begin(), work_buf.end());
    if (name != "data") {
        throw std::runtime_error(
            "expected a struct named data, but found one named " + name);
    }

    // Field name length
    nread += ReadDataElement(fp, work_buf, tag);
    const int32_t field_length = *reinterpret_cast<int32_t *>(&work_buf[0]);

    // Field names
    nread += ReadDataElement(fp, work_buf, tag);
    const int number_of_fields = work_buf.size() / field_length;

    std::vector<std::string> field_names;
    for (int i = 0; i < number_of_fields; i++) {
        const int start_ind = i * field_length;
        int length = 0;
        for (; length < field_length; length++) {
            if (work_buf[start_ind + length] == '\0') {
                break;
            }
        }
        field_names.push_back(
            std::string(work_buf.begin() + start_ind,
                        work_buf.begin() + start_ind + length));
    }

    nread += SkipPadding(fp, nread);

    int next_field_ind = 0;
    while (nread < remaining_bytes && next_field_ind < number_of_fields) {
        nread += ReadDataElementTag(fp, tag);

        if (tag.type != miMATRIX) {
            throw std::runtime_error("expected miMATRIX type, got " +
                                     std::to_string(tag.type));
        }

        // In the case that we find out this is another struct (e.g. the af
        // struct), we are just going to read the entire struct and skip it
        // since we don't use the data.
        const uint32_t main_tag_num_bytes = tag.num_bytes;
        const ssize_t main_tag_nread = nread;

        // Array flags
        nread += ReadDataElement(fp, work_buf, tag);
        if (tag.type != miUINT32) {
            throw std::runtime_error(
                "expected miUINT32 tag type for flags, got " +
                std::to_string(tag.type));
        }
        if (tag.num_bytes != 8) {
            throw std::runtime_error("expected 8 bytes for flags, got " +
                                     std::to_string(tag.num_bytes));
        }

        const uint32_t *const array_flags =
            reinterpret_cast<uint32_t *>(&work_buf[0]);

        const ClassType class_type = GetClassTypeFromArrayFlags(array_flags[0]);
        const bool isComplex = (array_flags[0] & 0x800) != 0;
        if (class_type == mxSTRUCT) {
            const ssize_t bytes_to_skip =
                main_tag_num_bytes - (nread - main_tag_nread);
            if (fseeko(fp, bytes_to_skip, SEEK_CUR) != 0) {
                throw std::runtime_error(
                    "failed to seek past nested struct element");
            }
            nread += bytes_to_skip;
            next_field_ind++;
            continue;
        }

        if (class_type != mxSINGLE) {
            throw std::runtime_error(
                "expected class type mxSINGLE for numeric data, got " +
                std::to_string(class_type));
            break;
        }

        // Matrix dimensions. We expect all matrices to be 2D. Note that 1D
        // arrays and scalars are still represented by 2D matrices (i.e. Nx1,
        // 1x1, etc.).
        nread += ReadDataElement(fp, work_buf, tag);
        const int num_dimensions = work_buf.size() / sizeof(int32_t);
        if (num_dimensions != 2) {
            throw std::runtime_error("expected two dimensional arrays, got " +
                                     std::to_string(num_dimensions) +
                                     "-D array");
        }
        std::vector<int32_t> dimensions(num_dimensions);
        const int32_t *const ptr = reinterpret_cast<int32_t *>(&work_buf[0]);
        for (int i = 0; i < num_dimensions; i++) {
            dimensions[i] = ptr[i];
        }

        // Array name. These are empty for struct fields because the struct
        // header includes all of the field names. The fields are then
        // concatenated in order and these names are left blank, presumably to
        // save space.
        nread += ReadDataElement(fp, work_buf, tag);

        // Real part of the numeric data
        nread += ReadDataElement(fp, real_data, tag);

        if (isComplex) {
            // Imaginary part of the numeric data
            nread += ReadDataElement(fp, imag_data, tag);
        }

        if (field_names[next_field_ind] == "fp") {
            // complex phase history
            block.num_freq = dimensions[0];
            block.num_pulses = dimensions[1];
            block.phase_history.resize(block.num_freq * block.num_pulses);
            const float *const rptr = reinterpret_cast<float *>(&real_data[0]);
            const float *const iptr = reinterpret_cast<float *>(&imag_data[0]);
            for (size_t i = 0; i < block.phase_history.size(); i++) {
                block.phase_history[i] = make_cuComplex(rptr[i], iptr[i]);
            }
        } else if (field_names[next_field_ind] == "freq") {
            const float *const rptr = reinterpret_cast<float *>(&real_data[0]);
            block.min_freq = rptr[0];
            block.del_freq = rptr[1] - rptr[0];
        } else if (field_names[next_field_ind] == "x") {
            const float *const rptr = reinterpret_cast<float *>(&real_data[0]);
            const size_t N = real_data.size() / sizeof(float);
            if (N > block.ant_pos.size()) {
                block.ant_pos.resize(N);
            }
            for (size_t i = 0; i < N; i++) {
                block.ant_pos[i].x = rptr[i];
            }
        } else if (field_names[next_field_ind] == "y") {
            const float *const rptr = reinterpret_cast<float *>(&real_data[0]);
            const size_t N = real_data.size() / sizeof(float);
            if (N > block.ant_pos.size()) {
                block.ant_pos.resize(N);
            }
            for (size_t i = 0; i < N; i++) {
                block.ant_pos[i].y = rptr[i];
            }
        } else if (field_names[next_field_ind] == "z") {
            const float *const rptr = reinterpret_cast<float *>(&real_data[0]);
            const size_t N = real_data.size() / sizeof(float);
            if (N > block.ant_pos.size()) {
                block.ant_pos.resize(N);
            }
            for (size_t i = 0; i < N; i++) {
                block.ant_pos[i].z = rptr[i];
            }
        }
        next_field_ind++;

        nread += SkipPadding(fp, nread);
    }
}

static void ReadDataFile(struct Gotcha::DataBlock &block,
                         const fs::path &file_path) {
    const auto filename = file_path.string();
    auto file_closer = [&filename](FILE *fp) {
        if (fp) {
            const int rc = fclose(fp);
            if (rc != 0) {
                std::cerr << "Failed to close " << filename << "; rc=" << rc
                          << "\n";
            }
        }
    };

    std::uintmax_t file_size = fs::file_size(file_path);

    std::unique_ptr<FILE, decltype(file_closer)> fp(
        fopen(filename.c_str(), "rb"), file_closer);
    if (fp.get() == nullptr) {
        throw std::runtime_error("failed to open " + filename);
    }

    DataHeader hdr;
    size_t nread = fread(&hdr, sizeof(DataHeader), 1, fp.get());
    if (nread != 1) {
        throw std::runtime_error("invalid read for " + filename +
                                 " data header");
    }

    // The Gotcha data is little-endian, so the ordering of the endianness
    // indicator bytes are MI (versus IM). We only handle the MI case.
    std::string endian_indicator;
    endian_indicator.push_back(
        static_cast<char>(((hdr.endian_indicator & 0xFF00) >> 8)));
    endian_indicator.push_back(
        static_cast<char>((hdr.endian_indicator & 0xFF)));
    if (endian_indicator != "MI") {
        throw std::runtime_error("unhandled endianness: got " +
                                 endian_indicator + ", expected MI");
    }

    const int64_t remaining_bytes = file_size - sizeof(hdr);
    ReadMatrixData(fp.get(), block, remaining_bytes);
}

std::string Gotcha::PolarizationToString(Gotcha::Polarization p) {
    switch (p) {
    case Gotcha::Polarization::HH:
        return std::string("HH");
    case Gotcha::Polarization::HV:
        return std::string("HV");
    case Gotcha::Polarization::VH:
        return std::string("VH");
    case Gotcha::Polarization::VV:
        return std::string("VV");
    }
    return std::string("");
}

SarReturnCode
Gotcha::Reader::ReadDataSet(struct Gotcha::DataBlock &data_set,
                            const struct Gotcha::DataRequest &request) {
    data_set.Reset();
    data_set.data_request = request;
    const std::string polarization =
        Gotcha::PolarizationToString(request.polarization);
    try {
        for (int i = request.first_az; i <= request.last_az; i++) {
            std::stringstream ss;
            ss << "pass" << request.pass << "/" << polarization
               << "/data_3dsar_pass" << request.pass << "_az"
               << std::setfill('0') << std::setw(3) << i << "_" << polarization
               << ".mat";
            const fs::path full_path = m_gotcha_dir + "/" + ss.str();
            LOG("Reading %s", full_path.string().c_str());
            Gotcha::DataBlock single_block;
            ReadDataFile(single_block, full_path);

            for (const auto ph : single_block.phase_history) {
                data_set.phase_history.push_back(ph);
            }
            for (const auto p : single_block.ant_pos) {
                data_set.ant_pos.push_back(p);
            }
            if (i == request.first_az) {
                data_set.num_freq = single_block.num_freq;
                data_set.min_freq = single_block.min_freq;
                data_set.del_freq = single_block.del_freq;
            }
            data_set.num_pulses += single_block.num_pulses;
        }
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        LOG_ERR("Exception reading data set: %s", e.what());
        return SarReturnCode::DataReadError;
    } catch (...) {
        std::cerr << "Caught unknown exception" << std::endl;
        LOG_ERR("Caught unknown exception reading data set");
        return SarReturnCode::DataReadError;
    }
    return SarReturnCode::Success;
}