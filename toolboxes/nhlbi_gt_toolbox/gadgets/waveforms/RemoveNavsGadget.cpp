#include <Node.h>
#include <mri_core_utility.h>
#include <mri_core_acquisition_bucket.h>
#include <ismrmrd/xml.h>
#include <gadgetron_mricore_export.h>
#include <mri_core_def.h>
#include <mri_core_data.h>
#include <hoNDArray_math.h>
#include <hoNDArray_utils.h>
#include <hoNDArray.h>
#include <ismrmrd/ismrmrd.h>

using namespace Gadgetron;
using namespace Gadgetron::Core;

class RemoveNavsGadget : public ChannelGadget<Core::Acquisition>
{

public:
    RemoveNavsGadget(const Core::Context &context, const Core::GadgetProperties &props) : ChannelGadget<Acquisition>(context, props)
    {
    }
    void process(InputChannel<Acquisition> &in, OutputChannel &out) override
    {
        for (auto message : in)
        {
            auto &[head, data, traj] = message;
            if ((head.isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_NAVIGATION_DATA) || head.isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_RTFEEDBACK_DATA)))
            {
                continue;
            }
            else
            {
                out.push(Core::Acquisition(std::move(head), std::move(data), std::move(traj)));
            }
            
        }
    }
};

GADGETRON_GADGET_EXPORT(RemoveNavsGadget)