/*
 * Spiral_Rovir.cpp
 *
 *      Author: Prakash Kumar
 */

#include <Node.h>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/range/combine.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/lambda/bind.hpp>

#include <boost/hana/functional/iterate.hpp>
#include <numeric>
#include <random>
#include <sstream>
#include <fstream>
#include <iostream>

#include <mri_core_grappa.h>
#include <vector_td_utilities.h>
#include <NonCartesianTools.h>
#include <NFFTOperator.h>
#include <cgSolver.h>
#include <hoNDArray_math.h>
#include <hoNDArray_utils.h>
#include <hoNDArray_elemwise.h>
#include <hoNFFT.h>
#include <hoNDFFT.h>
#include <GadgetronTimer.h>
#include <mri_core_coil_map_estimation.h>
#include <generic_recon_gadgets/GenericReconBase.h>
#include <ImageArraySendMixin.h>
#include <mri_core_kspace_filter.h>
#include <ImageIOBase.h>
#include <hoNDArray_fileio.h>
#include <cuNDArray_math.h>
#include <cuNDArray.h>

#include <iterator>
#include <hoNDArray_utils.h>
#include <hoCgSolver.h>
#include <hoNDImage_util.h>

#include "../spiral/SpiralBuffer.h"
#include "../utils/gpu/cuda_utils.h"
//#include <util_functions.h>
#include "../../utils/util_functions.h"
#include "noncartesian_reconstruction.h"

#include "reconParams.h"

#include <omp.h>
#include <algorithm>
#include <cmath>

using namespace Gadgetron;
using namespace Gadgetron::Core;
using namespace Gadgetron::Indexing;
using namespace nhlbi_toolbox::reconstruction;

class Spiral_Rovir : public ChannelGadget<Core::Acquisition>,
                          public ImageArraySendMixin<Spiral_Rovir>
{
public:
    std::vector<size_t> image_dims_;
    uint64d3 image_dims_os_;
    float oversampling_factor_;
    float kernel_width_;
    std::shared_ptr<SessionSpace> session_storage;
    bool verbose;

    // "global heap variables".... do they need to be "public"?
    //boost::shared_ptr<noncartesian_reconstruction<3>> reconstruction_;
    //boost::shared_ptr<cuNDArray<float_complext>> csm_;
    //boost::shared_ptr<cuNDArray<float>> dcf_;

    Spiral_Rovir(const Core::Context &context, const Core::GadgetProperties &props) : ChannelGadget<Core::Acquisition>(context, props), 
                                                                                     session_storage(context.storage.session)
    {
        kernel_width_ = 3;
        oversampling_factor_ = oversampling_factor;
        verbose = false;
    }

    void process(InputChannel<Core::Acquisition> &in,
                 OutputChannel &out) override
    {

        int selectedDevice = nhlbi_toolbox::utils::selectCudaDevice();
        unsigned int warp_size = cudaDeviceManager::Instance()->warp_size();
        size_t RO, E1, E2, CHA, N, S, SLC;
        nhlbi_toolbox::utils::enable_peeraccess();

        ISMRMRD::AcquisitionHeader acqhdr;

        Gadgetron::reconParams recon_params;

        size_t recon_idx = 0;
        size_t acq_count = 0;

        size_t out_counter = 0;

        auto n_tr_bin = ((header.encoding.at(0).encodingLimits.kspace_encoding_step_1.get().maximum + 1) * 
                        (header.encoding[0].encodingLimits.kspace_encoding_step_2.get().maximum + 1)) *
                        (header.encoding[0].encodingLimits.repetition.get().maximum + 1);
                        (header.encoding[0].encodingLimits.average.get().maximum + 1);

       GDEBUG_STREAM("kspace_encode_1_max" << (header.encoding.at(0).encodingLimits.kspace_encoding_step_1.get().maximum + 1));
       GDEBUG_STREAM("kspace_encode_2_max" << (header.encoding.at(0).encodingLimits.kspace_encoding_step_2.get().maximum + 1));
       //GDEBUG_STREAM("number of interleaves" << header.encoding.at(0).trajectoryDescription.get().userParameterLong[0].value);
       GDEBUG_STREAM("n_tr_bin" << n_tr_bin);

        std::vector<Core::Acquisition> allAcq(n_tr_bin);
        Gadgetron::hoNDArray<ISMRMRD::AcquisitionHeader> headers(n_tr_bin);

        std::vector<Core::Acquisition> csmAcq(n_tr_bin);
        Gadgetron::hoNDArray<ISMRMRD::AcquisitionHeader> csmHeaders(n_tr_bin);

        for (auto message : in)
        {
            if (acq_count < n_tr_bin)
            {
                auto &[head, data, traj] = message;
                acqhdr = head;
                csmAcq[acq_count] = std::move(message);
                csmHeaders[acq_count] = std::move(head);

                if (acq_count == (n_tr_bin - 1))
                {
                    auto acq_toRecon = csmAcq;
                    acq_toRecon.resize(n_tr_bin);
                    cudaSetDevice(selectedDevice);
                    auto &[headAcq, dataAcq, trajAcq] = acq_toRecon[0];
                    RO = dataAcq.get_size(0);
                    CHA = dataAcq.get_size(1);
                    E2 = this->header.encoding.front().encodedSpace.matrixSize.z;
                    N = dataAcq.get_size(3);
                    S = 1;
                    SLC = 1;

                    recon_params.numberChannels = CHA;
                    recon_params.RO = RO;
                    recon_params.ematrixSize = this->header.encoding.front().encodedSpace.matrixSize;
                    recon_params.rmatrixSize = this->header.encoding.front().encodedSpace.matrixSize; //this->header.encoding.front().reconSpace.matrixSize;

                    std::ostringstream encoded_matrix_ss;
                    encoded_matrix_ss << recon_params.ematrixSize.x << "_" << recon_params.ematrixSize.y << 
                                    "_" << recon_params.ematrixSize.z << "_" << CHA;
                    std::string str_ematrix = encoded_matrix_ss.str();
                    GDEBUG_STREAM("encoding matrix" << str_ematrix);

                    // for now, set the ematrix size to be a multiple of 32.
                    float matrix_overgrid = 2;
                    auto rmatrix = (int) (ceil(((float) recon_params.rmatrixSize.x * matrix_overgrid) / 32 ) * 32);
                    recon_params.ematrixSize.x = rmatrix;
                    recon_params.ematrixSize.y = rmatrix;
                    //recon_params.rmatrixSize.x = rmatrix;
                    //recon_params.rmatrixSize.y = rmatrix;

                    // TODO: PK revisit
                    // for now, set the rmatrix z size the same. This is because we want the full FOV for toeplitz (?)
		            recon_params.rmatrixSize.z = recon_params.ematrixSize.z;
                    recon_params.rmatrixSize.z = recon_params.ematrixSize.z;
                    recon_params.fov = this->header.encoding.front().encodedSpace.fieldOfView_mm;
                    recon_params.oversampling_factor_ = oversampling_factor_;
                    recon_params.kernel_width_ = kernel_width_;
                    recon_params.selectedDevice = selectedDevice;
                    recon_params.norm = 2;
                    recon_params.useIterativeDCWEstimated = false;
                    recon_params.oversampling_factor_dcf_ = oversampling_factor_; 
                    recon_params.kernel_width_dcf_ = 3; //3.2
                    recon_params.iterations_dcf = 10; // 10
                    this->initialize_encoding_space_limits(this->header);

                    auto reconstruction = noncartesian_reconstruction<3>(recon_params);

                    // reconstruct data.
                    auto [cuData, traj_csm, dcf_in] = reconstruction.organize_data(&acq_toRecon);

                    out_counter = 1;

                    std::ostringstream dcf_ss;
                    dcf_ss << "dcf_" << str_ematrix;
                    std::string dcf_str = dcf_ss.str();

                    std::ostringstream csm_ss;
                    csm_ss << "csm_" << str_ematrix;
                    std::string csm_str = csm_ss.str();

                    std::ostringstream csm_over_ss;
                    csm_over_ss << "csm_over_" << str_ematrix;
                    std::string csm_over_str = csm_over_ss.str();

                    std::ostringstream csm_image_ss;
                    csm_image_ss << "csm_image_" << str_ematrix;
                    std::string csm_image_str = csm_image_ss.str();

                    std::ostringstream csm_image_mc_ss;
                    csm_image_mc_ss << "csm_image_mc_" << str_ematrix;
                    std::string csm_image_mc_str = csm_image_mc_ss.str();

                    auto dcf = reconstruction.estimate_dcf(&traj_csm, &dcf_in);

                    square_inplace(&dcf);

                    auto dcf_cpu  = dcf.to_host();
                    //auto dcf_max  = dcf_cpu->at((size_t) std::floor((float)RO * 0.8));
                    //Gadgetron::clamp(dcf_cpu.get(), 0, dcf_max);

                    //auto dcf_max_array = Gadgetron::hoNDArray<float>(*dcf_cpu->get_dimensions().get());
                    //Gadgetron::fill(&dcf_max_array, Gadgetron::max(dcf_cpu.get()));
                    //Gadgetron::divide(dcf_cpu.get(), &dcf_max_array, dcf_cpu.get());
                    //auto scale = Gadgetron::max(dcf_cpu.get());

                    this->session_storage->store(dcf_str, *dcf_cpu);

                    cuNDArray<float_complext> channel_images(reconstruction.get_recon_dims());
                    {

                        reconstruction.reconstruct(&cuData, &channel_images, &traj_csm, &dcf);
                    }


                    cuData.clear();
                    traj_csm.clear();

                    auto csm_ = reconstruction.generateRoemerCSM(&channel_images);

                    // GDEBUG_STREAM("SAVING CSM to File....")
                    // // save the CSM for PYTHON
                    // std::ostringstream csm_ss_python;
                    // //csm_ss_python << "/opt/data/gt_data/csm_" << csm_->get_dimensions()->at(0) << "_" << csm_->get_dimensions()->at(1) << ".complex";
                    // csm_ss_python << "/opt/data/gt_data/csm_" << str_ematrix << ".complex";
                    // std::string buf_csm = csm_ss_python.str();
                    // nhlbi_toolbox::utils::write_gpu_nd_array(*csm_, buf_csm);



                    if (send_csm == 1)
                    {
                        auto csm_send =hoNDArray<std::complex<float>>(std::move(*boost::reinterpret_pointer_cast<hoNDArray<std::complex<float>>>(csm_->to_host()))); 
                        using namespace Gadgetron::Indexing;
                        IsmrmrdImageArray imarray_sense;
                        imarray_sense.data_ = csm_send;
                        nhlbi_toolbox::utils::attachHeadertoImageArray(imarray_sense, acqhdr, this->header);
                        prepare_image_array(imarray_sense, (size_t)0, ((int)series_counter), GADGETRON_IMAGE_GFACTOR);
                        imarray_sense.headers_(0, 0, 0).user_int[0]=head.idx.repetition;
                        imarray_sense.headers_(0, 0, 0).data_type = ISMRMRD::ISMRMRD_CXFLOAT;
                        imarray_sense.headers_(0, 0, 0).image_type = ISMRMRD::ISMRMRD_IMTYPE_COMPLEX;

                        out.push(imarray_sense);
                        series_counter++;
                    }

                    // save the multi-coil CSM image to the storage server. This is useful for ROVIR. 
                    // we also save it before cropping the dimensions.
                    auto csm_image_mc_cpu = std::move((channel_images).to_host());
                    this->session_storage->store(csm_image_mc_str, *csm_image_mc_cpu);

                    GDEBUG_STREAM("CSM estimation done. Sending images to storage server and/or next gadget....")
                    cuNDArray<float_complext> ci_cropped;
                    if (combine_csm == 0)
                        ci_cropped = channel_images;
                    else
                    {
                        channel_images *= *conj(csm_.get());

                        auto combined = sum(&channel_images, channel_images.get_number_of_dimensions() - 1);
                        
                        //HACK: update the supposedly private image_dims_ var (but it's public) so cropping actually does something.
                        reconstruction.image_dims_.at(0) = (int) recon_params.rmatrixSize.x * 1.5;
                        reconstruction.image_dims_.at(1) = (int) recon_params.rmatrixSize.y * 1.5;


                        ci_cropped = reconstruction.crop_to_recondims_centered(*combined);
                        (*combined).clear();
                        
                    }

                    // save the CSM for the storage server
                    auto csm_cpu = std::move((*csm_).to_host());
                    this->session_storage->store(csm_over_str, *csm_cpu);

                    auto csm_cropped = reconstruction.crop_to_recondims_centered(*csm_);
                    auto csm_cpu_cropped = std::move((csm_cropped).to_host());
                    this->session_storage->store(csm_str, *csm_cpu_cropped);

                    // // SAVE THE CSM IMAGE for python
                    // std::ostringstream csm_image_ss_python;
                    // //csm_image_ss_python << "/opt/data/gt_data/csm_image_" << csm_->get_dimensions()->at(0) << "_" << csm_->get_dimensions()->at(1) << ".complex";
                    // csm_image_ss_python << "/opt/data/gt_data/csm_image_" << str_ematrix << ".complex";
                    // std::string buf_csm_image = csm_image_ss_python.str();
                    // nhlbi_toolbox::utils::write_gpu_nd_array(ci_cropped, buf_csm_image);

                    // save the CSM image to the storage server
                    auto csm_image_cpu = std::move((ci_cropped).to_host());
                    this->session_storage->store(csm_image_str, *csm_image_cpu);
                    
                    channel_images.clear();

                    auto images = hoNDArray<std::complex<float>>(std::move(*boost::reinterpret_pointer_cast<hoNDArray<std::complex<float>>>(ci_cropped.to_host())));
                
                    using namespace Gadgetron::Indexing;
                    IsmrmrdImageArray image_out;
                    image_out.data_ = images;
                    nhlbi_toolbox::utils::attachHeadertoImageArray(image_out, acqhdr, this->header);
                    prepare_image_array(image_out, (size_t)0, ((int) series_counter), GADGETRON_IMAGE_REGULAR);

                    image_out.headers_(0, 0, 0).user_int[0]=head.idx.repetition;
                    image_out.headers_(0, 0, 0).data_type = ISMRMRD::ISMRMRD_CXFLOAT;
                    image_out.headers_(0, 0, 0).image_type = ISMRMRD::ISMRMRD_IMTYPE_COMPLEX;
                    image_out.acq_headers_ = csmHeaders;
                    GDEBUG_STREAM("ACQ HEADER EXPORT: " << image_out.acq_headers_->get_dimensions()->at(0));

                    out.push(image_out);
                }
            }
            acq_count ++;
        }
    }
protected:
    //NODE_PROPERTY(oversampling_factor, float, "oversampling_factor", 2.1);
    NODE_PROPERTY(oversampling_factor, float, "oversampling_factor", 2.1);
    NODE_PROPERTY(crop_begin, size_t, "crop_begin", 0);
    NODE_PROPERTY(crop_end, size_t, "crop_end", 0);
    NODE_PROPERTY(combine_csm, size_t, "combine_csm", 1);
    NODE_PROPERTY(send_csm, size_t, "send_csm", 0);
    int series_counter = 0;
};

GADGETRON_GADGET_EXPORT(Spiral_Rovir)
