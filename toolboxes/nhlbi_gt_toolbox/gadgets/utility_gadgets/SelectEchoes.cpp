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
#include <GadgetronTimer.h>

using namespace Gadgetron;
using namespace Gadgetron::Core;

class SelectEchoes : public ChannelGadget<Core::Acquisition>
{

public:
    SelectEchoes(const Core::Context &context, const Core::GadgetProperties &props) : ChannelGadget<Acquisition>(context, props)
    {
    }
    void process(InputChannel<Acquisition> &in, OutputChannel &out) override
    {
// #pragma omp parallel
// #pragma omp for
        {
        GadgetronTimer timer("Select echoes");
        for (auto message : in)
        {
            using namespace Gadgetron::Indexing;
            auto &[head, data, traj] = message;
            if((head.idx.set == setnum)){
                out.push(Core::Acquisition(std::move(head), std::move(data), std::move(traj)));
            }
                
        }
        }
    }
protected:
    NODE_PROPERTY(setnum, size_t, "start index to crop acquisition data", 0);
};

GADGETRON_GADGET_EXPORT(SelectEchoes)