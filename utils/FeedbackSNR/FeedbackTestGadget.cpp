/**
    \brief  Passes through an acquisition to the next gadget in the pipeline if the acquisition is below a certain time
*/
#include <gadgetron/Node.h>
#include <gadgetron/Gadget.h>
#include <gadgetron/hoNDArray.h>
#include <gadgetron/Types.h>
#include <ismrmrd/ismrmrd.h>
#include <ismrmrd/xml.h>
#include "../Feedback/FeedbackData.h"
using namespace Gadgetron;
using namespace Gadgetron::Core;

class FeedbackTestGadget : public ChannelGadget<Core::Acquisition> 
{

public:
    FeedbackTestGadget(const Core::Context &context, const Core::GadgetProperties &props) : ChannelGadget<Core::Acquisition>(context, props)
    {
    }
    void process(Core::InputChannel<Core::Acquisition>& in, Core::OutputChannel& out) override
    {
        auto i=0;
        for (auto message : in)
        {
            out.push(message);
            i++;
            if (i==ite_data){
                if (FB_data){
                    out.push(Gadgetron::FeedbackData{true, 10, 0,200.0f});
                }

                if (QC_data){
                    out.push(Gadgetron::QCPSFData{132.0f});
                }

                if (VENC_data){
                    out.push(Gadgetron::CalibratedVENCData{203});
                }
                
            }

        }
                
    }

protected:
NODE_PROPERTY(FB_data, bool, "Time in ms", true);
NODE_PROPERTY(QC_data, bool, "Ping Pong kz sampling ", false);
NODE_PROPERTY(VENC_data, bool, "Ping Pong kz sampling ", false);
NODE_PROPERTY(ite_data, long, "Ping Pong kz sampling ", 10);
};
GADGETRON_GADGET_EXPORT(FeedbackTestGadget)
