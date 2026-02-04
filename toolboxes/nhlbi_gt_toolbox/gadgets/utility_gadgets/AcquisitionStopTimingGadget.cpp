/**
    \brief  Passes through an acquisition to the next gadget in the pipeline if the acquisition is below a certain time
*/
#include <Node.h>
#include <Gadget.h>
#include <hoNDArray.h>
#include <Types.h>
#include <ismrmrd/ismrmrd.h>
#include <ismrmrd/xml.h>
using namespace Gadgetron;
using namespace Gadgetron::Core;

class AcquisitionStopTimingGadget : public ChannelGadget<Core::Acquisition> 
{

public:
    AcquisitionStopTimingGadget(const Core::Context &context, const Core::GadgetProperties &props) : ChannelGadget<Core::Acquisition>(context, props)
    {
    }
    void process(Core::InputChannel<Core::Acquisition>& in, Core::OutputChannel& out) override
    {
        bool time_limit_exceeded = false;
        auto idx = 0;
        uint32_t startTime=0;
        std::vector<size_t> info_to_send;
        auto time =0.0f;
        for (auto message : in)
        {
            if ((time_limit_exceeded)){
                continue;
            }         
                        
            auto &head = std::get<ISMRMRD::AcquisitionHeader>(message);

            if ((head.isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_NAVIGATION_DATA) || head.isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_RTFEEDBACK_DATA))){
                out.push(message);
                continue;
            }

            if(idx==0){
                startTime = head.acquisition_time_stamp;
            }
            
            time = float(head.acquisition_time_stamp-startTime)*2.5; // ms
            time_limit_exceeded=time > timit_limit;
            if ((time_limit_exceeded)){
                GDEBUG_STREAM("AcquisitionStopTimingGadget stopping acquisition at time: " << time << " ms and idx: " << idx);
                continue;
            }else{
                out.push(message);
            }
            
            idx++;   
        }
                
    }

protected:
NODE_PROPERTY(timit_limit, float, "Time in ms", 20000.0f);
};
GADGETRON_GADGET_EXPORT(AcquisitionStopTimingGadget)