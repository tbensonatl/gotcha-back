#include "common.h"

const char *GetSarReturnCodeString(SarReturnCode code) {
    switch (code) {
    case SarReturnCode::Success:
        return "success";
    case SarReturnCode::InvalidArg:
        return "invalid argument";
    case SarReturnCode::DataReadError:
        return "data read error";
    default:
        return "unknown";
    }
}

SarGpuKernel IntToSarGpuKernel(int kernel) {
    switch (kernel) {
    case static_cast<int>(SarGpuKernel::DoublePrecision):
        return SarGpuKernel::DoublePrecision;
    case static_cast<int>(SarGpuKernel::MixedPrecision):
        return SarGpuKernel::MixedPrecision;
    case static_cast<int>(SarGpuKernel::IncrPhaseLookup):
        return SarGpuKernel::IncrPhaseLookup;
    case static_cast<int>(SarGpuKernel::NewtonRaphsonTwoIter):
        return SarGpuKernel::NewtonRaphsonTwoIter;
    case static_cast<int>(SarGpuKernel::IncrRangeSmem):
        return SarGpuKernel::IncrRangeSmem;
    case static_cast<int>(SarGpuKernel::TextureSampling):
        return SarGpuKernel::TextureSampling;
    case static_cast<int>(SarGpuKernel::SinglePrecision):
        return SarGpuKernel::SinglePrecision;
    default:
        return SarGpuKernel::Invalid;
    }
}