#pragma once
#include <gadgetron/hoNDArray.h>
#include <ismrmrd/ismrmrd.h>

namespace Gadgetron
{
//MyFeedbackData
#pragma pack(push, 1) // 1-byte alignment
    struct FeedbackData
    {
        bool myBool;
        long myints[2];
        float myFloat;
    };
#pragma pack(pop) // Restore old alignment
//MyCalibratedVENC
#pragma pack(push, 1) // 1-byte alignment
    struct CalibratedVENCData
    {
        long myint;
    };
#pragma pack(pop) // Restore old alignment
//MyQCPSF
#pragma pack(push, 1) // 1-byte alignment
    struct QCPSFData
    {
        float myFloat;
    };
#pragma pack(pop) // Restore old alignment
}
