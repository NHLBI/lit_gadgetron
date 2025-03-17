#pragma once

#include <ismrmrd/ismrmrd.h>

#include <gadgetron/hoNDArray.h>

#include <gadgetron/Types.h>
#include <gadgetron/Writer.h>
#include "../FeedbackData.h"
namespace Gadgetron::Core::Writers {

    class FeedbackWriter
            : public Core::TypedWriter<Core::variant<Gadgetron::FeedbackData,Gadgetron::CalibratedVENCData,Gadgetron::QCPSFData>> {
    protected:
        void serialize(std::ostream &stream,const Core::variant<Gadgetron::FeedbackData,Gadgetron::CalibratedVENCData,Gadgetron::QCPSFData> &fdata);
    };
    
}