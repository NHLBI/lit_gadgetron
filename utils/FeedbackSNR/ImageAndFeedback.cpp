#include <gadgetron/Node.h>
#include <gadgetron/mri_core_utility.h>
#include <gadgetron/mri_core_acquisition_bucket.h>
#include <ismrmrd/xml.h>
#include <gadgetron/gadgetron_mricore_export.h>
#include <gadgetron/mri_core_def.h>
#include <gadgetron/mri_core_data.h>
#include <gadgetron/hoNDArray_math.h>
#include <gadgetron/hoNDArray_utils.h>
#include <gadgetron/hoNDArray.h>
#include <ismrmrd/ismrmrd.h>
#include <gadgetron/Types.h>
#include <gadgetron/GadgetMRIHeaders.h>
#include <gadgetron/GadgetronTimer.h>
#include "../Feedback/FeedbackData.h"
#include "../../utils/gpu/cuda_utils.h"
#include <util_functions.h>
#include <gadgetron/ImageArraySendMixin.h>
using namespace Gadgetron;
using namespace Gadgetron::Core;


class ImageAndFeedback : public ChannelGadget<Core::AnyImage>
{

public:
    ImageAndFeedback(const Core::Context &context, const Core::GadgetProperties &props) : ChannelGadget<Core::AnyImage>(context, props)
    {
    }
    void process(InputChannel<Core::AnyImage> &in, OutputChannel &out) override
    {
        GDEBUG_STREAM("-------------------------------------------TEST ---------------------------------------------------------");
        for (auto image : in) {
            if (holds_alternative<Core::Image<unsigned short>>(image))
            {
                auto &[header,mask,meta]=Core::get<Image<unsigned short>>(image);
                GDEBUG_STREAM("HEADER short" << header.user_int[1]);
                GDEBUG_STREAM("HEADER short Ufloat" << header.user_float[0]);
                if (VENC_data && inline_recon){
                    GDEBUG_STREAM("PushUVENC" << header.user_int[1]);
                    out.push(Gadgetron::CalibratedVENCData{header.user_int[1]});
                    out.push(image); 
                }
                if (QC_data && inline_recon ){
                    GDEBUG_STREAM("PushUflow" << header.user_float[0]);
                    out.push(Gadgetron::QCPSFData{header.user_float[0]});
                    continue;
                }

                if (SNR_data && inline_recon){
                    GDEBUG_STREAM("PushUSNR" << header.user_float[3]);
                    out.push(Gadgetron::FeedbackData{true, 0, 0,header.user_float[3]});
                    continue;
                }
            }

            if (holds_alternative<Core::Image<float>>(image))
            {
                auto &[header,mask,meta]=Core::get<Image<float>>(image);
                GDEBUG_STREAM("HEADER Float" << header.user_int[1]);
                /*
                if (VENC_data){
                    out.push(Gadgetron::CalibratedVENCData{header.user_int[1]});
                }
                if (QC_data){
                    out.push(Gadgetron::QCPSFData{header.user_float[0]});
                    continue;
                }
                */
            }

            if (holds_alternative<Core::Image<std::complex<float>>>(image))
            {
                auto &[header,mask,meta]=Core::get<Image<std::complex<float>>>(image);
                GDEBUG_STREAM("HEADER CXFloat" << header.user_int[1]);
                GDEBUG_STREAM("HEADER CXFloat Ufloat" << header.user_float[0]);
            }
            

            if (holds_alternative<Core::Image<std::complex<double>>>(image))
            {
                auto &[header,mask,meta]=Core::get<Image<std::complex<double>>>(image);
                GDEBUG_STREAM("HEADER CXDOUBLE" << header.user_int[1]);
            }
            
            if (inline_recon){
                GDEBUG_STREAM("Not sending images");
                //out.push(image); 
            }
            else{
                out.push(image); 
            }
                   


        }
        
    }

protected:
NODE_PROPERTY(QC_data, bool, "Ping Pong kz sampling ", false);
NODE_PROPERTY(VENC_data, bool, "Ping Pong kz sampling ", false);
NODE_PROPERTY(SNR_data, bool, "Ping Pong kz sampling ", false);
NODE_PROPERTY(inline_recon, bool, "Ping Pong kz sampling ", false);
};

GADGETRON_GADGET_EXPORT(ImageAndFeedback)