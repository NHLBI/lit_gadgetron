
#include "io/primitives.h"
#include "mri_core_data.h"


#include "FeedbackReader.h"

#include <ismrmrd/waveform.h>
#include "../FeedbackData.h"
namespace Gadgetron::Core::Readers {

    Core::Message FeedbackReader::read(std::istream& stream) {

        using namespace Core;
        using namespace std::literals;

        //long myints[2];

        auto fdata = IO::read<FeedbackData>(stream);
        return Message(fdata);
    }
    uint16_t FeedbackReader::slot() {
        return MessageID::GADGET_MESSAGE_ISMRMRD_FEEDBACK;
    }

    

    GADGETRON_READER_EXPORT(FeedbackReader)
}