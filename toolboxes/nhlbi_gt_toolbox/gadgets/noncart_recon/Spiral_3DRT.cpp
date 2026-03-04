/*
 * SpiralMocoRecon.cpp
 *
 *  Created on: September 17th, 2021
 *      Author: Ahsan Javed
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

class Spiral_3DRT : public ChannelGadget<Core::Acquisition>,
                          public ImageArraySendMixin<Spiral_3DRT>
{
public:
    std::vector<size_t> image_dims_;
    uint64d3 image_dims_os_;
    float oversampling_factor_;
    float kernel_width_;
    std::shared_ptr<SessionSpace> session_storage;
    bool verbose;

    // "global heap variables".... do they need to be "public"?
    boost::shared_ptr<noncartesian_reconstruction<3>> reconstruction_;
    boost::shared_ptr<cuNDArray<float_complext>> csm_;
    boost::shared_ptr<cuNDArray<float>> dcf_;

    Spiral_3DRT(const Core::Context &context, const Core::GadgetProperties &props) : ChannelGadget<Core::Acquisition>(context, props), 
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

        auto n_tr_bin = repetitions;
        if (n_tr_bin == 0)
        {
            n_tr_bin = ((header.encoding.at(0).trajectoryDescription.get().userParameterLong[0].value) *
                        (header.encoding[0].encodingLimits.kspace_encoding_step_2.get().maximum + 1));
        }
        GDEBUG_STREAM("kspace_encode_1_max" << (header.encoding.at(0).encodingLimits.kspace_encoding_step_1.get().maximum + 1));
        GDEBUG_STREAM("kspace_encode_2_max" << (header.encoding.at(0).encodingLimits.kspace_encoding_step_2.get().maximum + 1));
        GDEBUG_STREAM("number of interleaves" << header.encoding.at(0).trajectoryDescription.get().userParameterLong[0].value);
        GDEBUG_STREAM("n_tr_bin" << n_tr_bin);

        std::vector<Core::Acquisition> allAcq(n_tr_bin);
        Gadgetron::hoNDArray<ISMRMRD::AcquisitionHeader> headers(n_tr_bin);

        for (auto message : in)
        {
            auto &[head, data, traj] = message;
            acqhdr = head;

            allAcq[recon_idx] = std::move(message);
            headers[recon_idx] = std::move(head);
            recon_idx ++;
            acq_count ++;

            if (acq_count == n_tr_bin)
            {
                // this is the first time we are fully sampled, so we should get reconstruction, dcf, and csm.
                // try to get them from the storage server, but if it fails.... generate anew.

                auto acq_toRecon = allAcq;
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
                recon_params.rmatrixSize = this->header.encoding.front().reconSpace.matrixSize;

                std::ostringstream encoded_matrix_ss;
                encoded_matrix_ss << recon_params.ematrixSize.x << "_" << recon_params.ematrixSize.y << 
                                "_" << recon_params.ematrixSize.z << "_" << CHA;
                std::string str_ematrix = encoded_matrix_ss.str();
                GDEBUG_STREAM("encoding matrix" << str_ematrix);

                // for now, set the rmatrix size to be a multiple of 32.
                auto rmatrix = (int) (round(((float) recon_params.rmatrixSize.x) / 32 ) * 32);
                recon_params.ematrixSize.x = rmatrix;
                recon_params.ematrixSize.y = rmatrix;
                recon_params.rmatrixSize.x = rmatrix;
                recon_params.rmatrixSize.y = rmatrix;


                GDEBUG_STREAM("recon_params.ematrixSize.x" << recon_params.ematrixSize.x);
                GDEBUG_STREAM("recon_params.ematrixSize.y" << recon_params.ematrixSize.y);
                GDEBUG_STREAM("recon_params.ematrixSize.z" << recon_params.ematrixSize.z);
                GDEBUG_STREAM("recon_params.rmatrixSize.x" << recon_params.rmatrixSize.x);
                GDEBUG_STREAM("recon_params.rmatrixSize.y" << recon_params.rmatrixSize.y);
                GDEBUG_STREAM("recon_params.rmatrixSize.z" << recon_params.rmatrixSize.z);

                // TODO: PK revisit
                // for now, set the rmatrix z size the same. This is because we want the full FOV for toeplitz (?)
                recon_params.rmatrixSize.z = recon_params.ematrixSize.z;
                recon_params.fov = this->header.encoding.front().encodedSpace.fieldOfView_mm;
                recon_params.oversampling_factor_ = oversampling_factor_;
                recon_params.kernel_width_ = kernel_width_;
                recon_params.selectedDevice = selectedDevice;
                recon_params.norm = 2;
                recon_params.useIterativeDCWEstimated = true;
                recon_params.oversampling_factor_dcf_ = oversampling_factor_; 
                recon_params.kernel_width_dcf_ = 5.5;
                recon_params.iterations_dcf = 20;
                this->initialize_encoding_space_limits(this->header);

                reconstruction_ = boost::make_shared<noncartesian_reconstruction<3>>(recon_params);

                auto reconstruction = *reconstruction_;

                // reconstruct data.
                auto [cuData, traj_csm, dcf_in] = reconstruction.organize_data(&acq_toRecon);

                //if (out_counter == 0)
                //    nhlbi_toolbox::utils::write_gpu_nd_array(cuData, "/opt/data/gt_data/kspace_0.complex");

                std::ostringstream dcf_ss;
                //dcf_ss << "dcf_" << recon_params.ematrixSize.x << "_" << recon_params.ematrixSize.y;
                dcf_ss << "dcf_" << str_ematrix;
                std::string dcf_str = dcf_ss.str();

                std::ostringstream csm_ss;
                //csm_ss << "csm_" << recon_params.ematrixSize.x << "_" << recon_params.ematrixSize.y;
                csm_ss << "csm_" << str_ematrix;
                std::string csm_str = csm_ss.str();

                // Load the DCF.
                try
                {
                    auto dcf_cpu = this->session_storage->get_latest<hoNDArray<float>>(dcf_str);

                    //PK funny: actually this debug message makes the try block fail gracefully
                    // TODO: figure out how to identify a 404 error and skip out gracefully, without
                    // having to resort to checking the dimensions to fail.
                    GDEBUG_STREAM("DCF did it fail? size: " << dcf_cpu->get_dimensions()->at(0));

                    nhlbi_toolbox::utils::write_cpu_nd_array(*dcf_cpu, "/opt/data/gt_data/dcf_scaled.real");

                    auto dcf_resize = hoNDArray<float>(dcf_in.get_dimensions());
                    for (size_t dcf_i = 0; dcf_i < dcf_resize.get_dimensions()->at(0); dcf_i++){
                        dcf_resize.at(dcf_i) = dcf_cpu->at(dcf_i);
                    }
                    dcf_ = boost::make_shared<cuNDArray<float>>(cuNDArray<float>(dcf_resize));
                }
                catch (...)
                {
                    //gen dcf if we can't load -> safety to regenerate!
                    GWARN_STREAM("Warning: DCF is being estimated from this scan, instead of loaded separately....");
                    auto dcf = reconstruction.estimate_dcf(&traj_csm, &dcf_in); 
                    dcf_ = boost::make_shared<cuNDArray<float>>(cuNDArray<float>(dcf));
                }

                // Load the CSM.
                try
                {
                    auto csm_cpu = this->session_storage->get_latest<hoNDArray<float_complext>>(csm_str);
                    csm_ = boost::make_shared<cuNDArray<float_complext>>(cuNDArray<float_complext>(*csm_cpu));
                }
                catch (...)
                {
                    //gen csm if we can't load -> safety. But it could be undersampled.
                    GWARN_STREAM("Warning: CSM is being estimated from this scan, instead of loaded separately....");
                    cuNDArray<float_complext> channel_images(reconstruction.get_recon_dims());
                    {
                        reconstruction.reconstruct(&cuData, &channel_images, &traj_csm, dcf_.get());
                    }
                    csm_ = reconstruction.generateRoemerCSM(&channel_images);
                    cuData.clear();
                    traj_csm.clear();
                }

                // sqrt DCF if needed.
                if (sqrt_dcf)
                    Gadgetron::sqrt_inplace(dcf_.get());

            }

            if (recon_idx == n_tr_bin)
            {
                //re-set the recon_idx. We want to generate images with n_tr_bin over and over until acq_end.
                recon_idx = 0;
                out.push(allAcq);
                GadgetronTimer timer_all("recon pipeline timer", false);
                timer_all.start();

                auto acq_toRecon = allAcq;
                acq_toRecon.resize(n_tr_bin);
                cudaSetDevice(selectedDevice);
                auto &[headAcq, dataAcq, trajAcq] = acq_toRecon[0];
                
                auto reconstruction = *reconstruction_;
                
                GadgetronTimer timer_organize("organize_data", false);
                timer_organize.start();
                auto [cuData, traj, dcf_in] = reconstruction.organize_data(&acq_toRecon);
                timer_organize.stop();

                GadgetronTimer timer_dcf("dcf", false);
                timer_dcf.start();
                auto dcf = *dcf_;
                timer_dcf.stop();

                cuNDArray<float_complext> channel_images(reconstruction.get_recon_dims());

                {
                    GadgetronTimer recon_timer("reconstruction", false);
                    recon_timer.start();
                    reconstruction.reconstruct(&cuData, &channel_images, &traj, &dcf);
                    cudaDeviceSynchronize();
                    recon_timer.stop();
                }

                // save the data -- there must be a better way (I hope)
                if (out_counter == 0) 
                { 
                    std::ostringstream oss;
                    oss << "/opt/data/gt_data/kspace_" << out_counter << ".complex";
                    std::string buf = oss.str();
                    nhlbi_toolbox::utils::write_gpu_nd_array(cuData, buf);
                }
                    
                out_counter = out_counter + 1;

                GadgetronTimer coil_comb("coil_comb", false);
                coil_comb.start();
                cuNDArray<float_complext> ci_cropped;
                if (combine_csm == 0)
                    ci_cropped = channel_images;
                else
                {
                    channel_images *= *conj(csm_.get());

                    auto combined = sum(&channel_images, channel_images.get_number_of_dimensions() - 1);
                
                    ci_cropped = reconstruction.crop_to_recondims(*combined);

                    (*combined).clear();

                }
                channel_images.clear();
                coil_comb.stop();                

                GadgetronTimer finish("finish", false);
                finish.start();
                auto images = hoNDArray<std::complex<float>>(std::move(*boost::reinterpret_pointer_cast<hoNDArray<std::complex<float>>>(ci_cropped.to_host())));
            
                using namespace Gadgetron::Indexing;
                IsmrmrdImageArray imarray_sense;
                imarray_sense.data_ = images;
                nhlbi_toolbox::utils::attachHeadertoImageArray(imarray_sense, acqhdr, this->header);
                prepare_image_array(imarray_sense, (size_t)0, ((int) series_counter), GADGETRON_IMAGE_REGULAR);

                timer_all.stop();
                finish.stop();

                imarray_sense.headers_(0, 0, 0).user_int[0]=head.idx.repetition;
                imarray_sense.headers_(0, 0, 0).data_type = ISMRMRD::ISMRMRD_CXFLOAT;
                imarray_sense.headers_(0, 0, 0).image_type = ISMRMRD::ISMRMRD_IMTYPE_COMPLEX;
                imarray_sense.acq_headers_ = headers;


                out.push(imarray_sense);
            }
        }
    }

protected:
    NODE_PROPERTY(oversampling_factor, float, "oversampling_factor", 1.25);
    NODE_PROPERTY(crop_begin, size_t, "crop_begin", 0);
    NODE_PROPERTY(crop_end, size_t, "crop_end", 0);
    NODE_PROPERTY(repetitions, size_t, "repetitions", 0);
    NODE_PROPERTY(combine_csm, size_t, "combine_csm", 1);
    NODE_PROPERTY(send_csm, size_t, "send_csm", 0);
    NODE_PROPERTY(sqrt_dcf, size_t, "sqrt_dcf", 0);
    int series_counter = 0;
};

GADGETRON_GADGET_EXPORT(Spiral_3DRT)