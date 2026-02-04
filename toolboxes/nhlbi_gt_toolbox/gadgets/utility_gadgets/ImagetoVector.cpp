
#include <GadgetMRIHeaders.h>
#include <Node.h>
#include <Types.h>
#include <hoNDArray.h>
#include <hoNDArray_math.h>
#include <mri_core_data.h>
#include <gadgetron_mricore_export.h>
#include "hoNDArray_utils.h"

class ImagetoVector : public Core::ChannelGadget<Core::Image<double>>
{
public:
    using Core::ChannelGadget<Core::Image<double>>::ChannelGadget;

    std::vector<std::vector<std::vector<std::vector<size_t>>>> ImagetoVector_convert(hoNDArray<double> data)
    {
        GDEBUG_STREAM("Converting image to vector");
        std::vector<std::vector<std::vector<std::vector<size_t>>>> indices;
        for (auto ne = 0; ne < data.get_size(3); ne++)
        {   std::vector<std::vector<std::vector<size_t>>> indices_0;
            for (auto nc = 0; nc < data.get_size(2); nc++)
            {   std::vector<std::vector<size_t>> indices_1;
                for (auto nr = 0; nr < data.get_size(1); nr++)
                {
                    GDEBUG_STREAM("data(" << ne << "," << nc << "," << nr << ") = " << data(0,nr, nc,ne));
                    auto stride = data.get_size(0) * nr + data.get_size(0)*data.get_size(1) * nc + data.get_size(0)*data.get_size(1)*data.get_size(2) * ne;
                    std::vector<size_t> indices_2(data.get_data_ptr() + 1 + stride, data.get_data_ptr() + size_t(1 + data(0,nr, nc,ne)) + stride);
                    indices_1.push_back(indices_2);
                }
                indices_0.push_back(indices_1);
            }   
            indices.push_back(indices_0);
        }
        GDEBUG_STREAM("NE: " << indices.size());
        GDEBUG_STREAM("NC: " << indices[0].size());
        GDEBUG_STREAM("NR: " << indices[0][0].size());

        return indices;
    }
    void process(Core::InputChannel<Core::Image<double>> &in, Core::OutputChannel &out) override
    {
        std::vector<hoNDArray<double>> images;
        std::vector<size_t> images_index;
        for (auto message : in)
        {
            auto &[imhead, data, meta] = message;
            // visit([&](auto message)
            //       { splitInputData(message, out); },
            //       msg);
            auto pdata = permute(data, {3, 2, 1, 0}); // Assuming data is in the format [ne,nc,nr,ni] - > [ni,nr,nc,ne]
            out.push(ImagetoVector_convert(pdata));
            // images.push_back(data);
            // images_index.push_back(imhead.image_index);
        }

        // std::vector<std::vector<size_t>> idx_to_send;
        // for (auto msg : images)
        // {
        //     idx_to_send.push_back(ImagetoVector_convert(msg));
        // }
        // GDEBUG_STREAM("idx_to_send: " << idx_to_send.size());
        // GDEBUG_STREAM("idx_to_send[0]: " << idx_to_send[0].size());

        // out.push(idx_to_send);
    }
};
GADGETRON_GADGET_EXPORT(ImagetoVector)
