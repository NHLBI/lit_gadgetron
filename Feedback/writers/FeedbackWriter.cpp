#include "FeedbackWriter.h"
#include "io/primitives.h"
#include "../MessageID.h"
#include "../FeedbackData.h"

namespace Gadgetron::Core::Writers {
        
        void FeedbackWriter::serialize(std::ostream &stream,const Core::variant<Gadgetron::FeedbackData,Gadgetron::CalibratedVENCData,Gadgetron::QCPSFData> &fdata){
                
                IO::write(stream, GADGET_MESSAGE_ISMRMRD_FEEDBACK);
                if (holds_alternative<Gadgetron::FeedbackData>(fdata))
                {
                        IO::write(stream,uint32_t(sizeof("MyFeedback")));
                        IO::write(stream,"MyFeedback");
                        
                }
                if (holds_alternative<Gadgetron::CalibratedVENCData>(fdata))
                {
                        IO::write(stream,uint32_t(sizeof("MyCalibratedVENC")));
                        IO::write(stream,"MyCalibratedVENC");
                        
                }
                if (holds_alternative<Gadgetron::QCPSFData>(fdata))
                {
                        IO::write(stream,uint32_t(sizeof("MyQCPSF")));
                        IO::write(stream,"MyQCPSF");
                        
                }
                IO::write(stream,uint32_t(sizeof(fdata)));
                IO::write(stream,fdata);

        }
        GADGETRON_WRITER_EXPORT(FeedbackWriter)


}

