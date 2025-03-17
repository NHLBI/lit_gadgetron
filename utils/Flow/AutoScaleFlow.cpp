#include "PureGadget.h"
#include "Types.h"
#include "hoNDArray_math.h"
#include <algorithm>

using namespace Gadgetron;
using namespace Gadgetron::Core;
    template <typename T>
    Image<T> autoscale(Image<T> &image) {
        auto &header = std::get<ISMRMRD::ImageHeader>(image);
        auto &data = std::get<hoNDArray<T>>(image);
        auto &meta = std::get<optional<ISMRMRD::MetaContainer>>(image);	

        if (header.image_type == ISMRMRD::ISMRMRD_IMTYPE_MAGNITUDE) { 
            auto current_scale_=100;
            GDEBUG_STREAM("Scale 100");
            for (auto& d : data){
                d *= current_scale_;
            }
        }
        if (header.image_type == ISMRMRD::ISMRMRD_IMTYPE_PHASE) { //scale Phase
            for (auto& d : data){
                d = ((d/M_PI)+1)*2048;
            }
        }
        return Image<T>(header,data,meta);
    }
    template<class T>
    Image<std::complex<T>> autoscale(Image<std::complex<T>> &image) {GERROR("Autoscaling image is not well defined for complex images. ");
    return image;}

    class AutoScaleFlow : public PureGadget<Core::AnyImage, Core::AnyImage>
    {

    public:

        AutoScaleFlow(const Core::Context &context, const Core::GadgetProperties &props) : PureGadget<Core::AnyImage, Core::AnyImage>(context, props)
    {
    }
        
        AnyImage process_function(AnyImage image) const override{
            return visit([&](auto &image) -> AnyImage { return autoscale(image); }, image);
        
        }
        
    protected:;
    }; 

GADGETRON_GADGET_EXPORT(AutoScaleFlow)