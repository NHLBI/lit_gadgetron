
/**
    \brief  Passes through an acquisition to the next gadget in the pipeline if the acquisition is below a certain time
*/
#include <gadgetron/Node.h>
#include <gadgetron/Gadget.h>
#include <gadgetron/hoNDArray.h>
#include <gadgetron/Types.h>
#include <ismrmrd/ismrmrd.h>
#include <ismrmrd/xml.h>
#include <gadgetron/Node.h>
#include <gadgetron/gadgetron_mricore_export.h>
#include <gadgetron/hoNDArray.h>
#include <gadgetron/mri_core_acquisition_bucket.h>
#include <gadgetron/mri_core_data.h>
#include <complex>
#include <ismrmrd/ismrmrd.h>
#include <ismrmrd/xml.h>
#include <gadgetron/hoNDArray_math.h>
#include <gadgetron/mri_core_utility.h>
#include <gadgetron/hoNDFFT.h>
#include <util_functions.h>

using namespace Gadgetron;
using namespace Gadgetron::Core;

class AverageReconGadget : public ChannelGadget<IsmrmrdReconData>
{

public:
    AverageReconGadget(const Core::Context &context, const Core::GadgetProperties &props) : ChannelGadget<IsmrmrdReconData>(context, props)
    {
    }
    void process(Core::InputChannel<IsmrmrdReconData>& input, Core::OutputChannel& out) override
    {
        size_t chunk_N=0;
        for (IsmrmrdReconData reconData : input) {

            GDEBUG_STREAM("TEST");
            // Iterate over all the recon bits
            for (std::vector<IsmrmrdReconBit>::iterator it = reconData.rbit_.begin(); it != reconData.rbit_.end(); ++it) {
                // Grab a reference to the buffer containing the imaging data
                // We are ignoring the reference data
                IsmrmrdDataBuffered& dbuff = it->data_;
                hoNDArray< std::complex<float> >& data = dbuff.data_;
                hoNDArray< std::complex<float> > data_chunk;
                hoNDArray< std::complex<float> > data_chunk_S;
                hoNDArray< std::complex<float> > data_average;
                // Data 7D, fixed order [E0, E1, E2, CHA, N, S, LOC]
                uint16_t E0 = dbuff.data_.get_size(0);
                uint16_t E1 = dbuff.data_.get_size(1);
                uint16_t E2 = dbuff.data_.get_size(2);
                uint16_t CHA = dbuff.data_.get_size(3);
                uint16_t N = dbuff.data_.get_size(4);
                uint16_t S = dbuff.data_.get_size(5);
                uint16_t LOC = dbuff.data_.get_size(6);
                bool count_sampling_freq = true;
                GDEBUG_STREAM("E0: " << E0 << "; E1: " << E1 << "; E2: " << E2 << "; CHA: " << CHA << "; N: " << N << "; S: " << S << "; LOC: " << LOC );
                nhlbi_toolbox::utils::chunk_data_S(data, chunk_S_bool,chunk_S_input, data_chunk_S);
                GDEBUG_STREAM("Chunk S : N: " << data_chunk_S.get_size(4) << " S: " << data_chunk_S.get_size(5))

                nhlbi_toolbox::utils::chunk_data_N(data_chunk_S, chunk_N_bool,chunk_N, data_chunk);
                chunk_N = data_chunk.get_size(4);
                GDEBUG_STREAM("Chunk N : N: " << data_chunk.get_size(4) << " S: " << data_chunk.get_size(5))

                Gadgetron::compute_averaged_data_N_S(data_chunk, average_all_N, average_all_S, count_sampling_freq, data_average);
                uint16_t scale_S=data_chunk.get_size(5);
                Gadgetron::scal((float)(sqrt(scale_S)), data_average);
                uint16_t NN=data_average.get_size(4);
                uint16_t NS=data_average.get_size(5);
                GDEBUG_STREAM("Average N/S : N: " << NN << " S: " << NS)
                hoNDArray<ISMRMRD::AcquisitionHeader> new_header = hoNDArray<ISMRMRD::AcquisitionHeader>(E1, E2, NN, NS, LOC);

                for (auto n=0; n<new_header.get_number_of_elements(); n++)
                {
                    new_header(n) = dbuff.headers_(n);
                }
                dbuff.headers_=new_header;
                dbuff.data_ = data_average;  
                
            }
            out.push(reconData);
        }
        
                
    }

protected:
NODE_PROPERTY(average_all_N, bool, "Whether to average all N for data generation", false);
NODE_PROPERTY(average_all_S, bool, "Whether to average all S for data generation", false);
NODE_PROPERTY(chunk_N_bool, bool, "Whether to chunk N", true);
NODE_PROPERTY(chunk_S_bool, bool, "Whether to chunk S", false);
NODE_PROPERTY(chunk_S_input, uint16_t, "Chunk_S_input Only for retro recon", 1);
};
GADGETRON_GADGET_EXPORT(AverageReconGadget)
