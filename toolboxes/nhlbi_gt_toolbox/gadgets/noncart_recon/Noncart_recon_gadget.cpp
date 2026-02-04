/*
 * Noncart_recon_gadget.cpp
 *
 *  Created on: September 17th, 2021
 *      Author: Ahsan Javed
 */

#include "../spiral/SpiralBuffer.h"
#include "cuNonCartesianMOCOOperator.h"
#include "cuNonCartesianTSenseOperator.h"
#include "mri_concomitant_field_correction_lit.h"
#include <GadgetronTimer.h>
#include <ImageArraySendMixin.h>
#include <ImageIOBase.h>
#include <NFFTOperator.h>
#include <Node.h>
#include <NonCartesianTools.h>
#include <b1_map.h>
#include <boost/hana/functional/iterate.hpp>
#include <boost/lambda/bind.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/make_shared.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/range/combine.hpp>
#include <boost/shared_ptr.hpp>
#include <cgSolver.h>
#include <cuCgPreconditioner.h>
#include <cuCgSolver.h>
#include <cuImageOperator.h>
#include <cuNDArray.h>
#include <cuNDArray_converter.h>
#include <cuNDArray_math.h>
#include <cuNDArray_operators.h>
#include <cuNDFFT.h>
#include <cuNFFT.h>
#include <cuNlcgSolver.h>
#include <cuNonCartesianSenseOperator.h>
#include <cuSDC.h>
#include <cuTvOperator.h>
#include <cuTvPicsOperator.h>
#include <fstream>
#include <generic_recon_gadgets/GenericReconBase.h>
#include <hoCgSolver.h>
#include <hoNDArray_elemwise.h>
#include <hoNDArray_fileio.h>
#include <hoNDArray_math.h>
#include <hoNDArray_utils.h>
#include <hoNDFFT.h>
#include <hoNDImage_util.h>
#include <hoNFFT.h>
#include <iostream>
#include <iterator>
#include <mri_core_coil_map_estimation.h>
#include <mri_core_grappa.h>
#include <mri_core_kspace_filter.h>
#include <numeric>
#include <omp.h>
#include <random>
#include <sstream>
#include <vector_td_utilities.h>
// #include "../utils/gpu/cuda_utils.h"
// #include "/usr/local/include/cuTVPrimalDualOperator.h"

#include "SpiralReconBuffer.h"
#include "cuGpBbSolver.h"
#include "cuPartialDerivativeOperator.h"
#include "cuPartialDerivativeOperator2.h"
#include "cuSbcCgSolver.h"
#include "densityCompensation.h"
#include "noncartesian_reconstruction.h"
#include "noncartesian_reconstruction_3D.h"
#include "noncartesian_reconstruction_4D.h"
#include "noncartesian_reconstruction_5D.h"
#include "reconParams.h"
#include <algorithm>
#include <cmath>
#include <curand.h>
#include <omp.h>
#include <python_toolbox.h>
#include <util_functions.h>

using namespace Gadgetron;
using namespace Gadgetron::Core;
using namespace Gadgetron::Indexing;

class Noncart_recon_gadget
    : public ChannelGadget<Core::variant<Core::Acquisition, Gadgetron::reconParams,
                                         std::vector<std::vector<std::vector<std::vector<size_t>>>>>>,
      public ImageArraySendMixin<Noncart_recon_gadget> {
  public:
    std::vector<size_t> image_dims_;
    uint64d3 image_dims_os_;
    bool verbose;
    bool csm_calculated_ = false;

    float lambda_spatial_;
    float lambda_time_;
    std::filesystem::path python_path;

    Noncart_recon_gadget(const Core::Context& context, const Core::GadgetProperties& props)
        : ChannelGadget<Core::variant<Core::Acquisition, Gadgetron::reconParams,
                                      std::vector<std::vector<std::vector<std::vector<size_t>>>>>>(context, props) {
        verbose = false;

        const char* conda_prefix = std::getenv("CONDA_PREFIX");
        if (conda_prefix) {
            python_path = std::filesystem::path(conda_prefix) / "share" / "gadgetron" / "python";
        } else {
            python_path = std::filesystem::path("/opt/conda/envs/gadgetron") / "share" / "gadgetron" / "python";
        }
        Gadgetron::initialize_python();
        Gadgetron::add_python_path(python_path.generic_string());
    }

    void process(InputChannel<Core::variant<Core::Acquisition, Gadgetron::reconParams,
                                            std::vector<std::vector<std::vector<std::vector<size_t>>>>>>& in,
                 OutputChannel& out) override {

        PythonFunction<hoNDArray<float>> register_images("registration_gadget_call", "registration_images");

        auto matrixSize = this->header.encoding.front().reconSpace.matrixSize;
        auto fov = this->header.encoding.front().reconSpace.fieldOfView_mm;
        auto resx = fov.x / float(matrixSize.x);
        auto resy = fov.y / float(matrixSize.y);
        auto resz = fov.z / float(matrixSize.z);
        unsigned int warp_size = cudaDeviceManager::Instance()->warp_size();
        size_t RO, E1, E2, CHA, N, S, SLC;
        // nhlbi_toolbox::utils::enable_peeraccess();

        hoNDArray<float> trajectories;

        std::tuple<boost::shared_ptr<hoNDArray<floatd3>>, boost::shared_ptr<hoNDArray<float>>> traj_dcw;
        IsmrmrdDataBuffered* buffer;
        IsmrmrdImageArray imarray;
        ISMRMRD::AcquisitionHeader acqhdr;
        hoNDArray<size_t> shots_per_time;
        // hoNDArray<size_t> shots_per_time_respiratory, shots_per_time_cardiac;
        boost::shared_ptr<cuNDArray<float_complext>> csm, channel_images;
        hoNDArray<floatd3> gradients;

        Gadgetron::reconParams recon_params;
        Gadgetron::reconParams recon_params_adv;

        // Declare vectors for temporalTV
        std::vector<hoNDArray<float_complext>> dataVector;
        std::vector<hoNDArray<float>> dcwVector;
        std::vector<hoNDArray<floatd3>> trajVector;
        std::vector<cuNDArray<float>> sctVector;
        boost::shared_ptr<cuNDArray<float_complext>> cweights;
        arma::fvec freq_bins;

        using namespace Gadgetron::Indexing;

        auto kspace_scaling = 1e-3 * fov.x / matrixSize.x;
        GDEBUG_STREAM("matrix z: " << matrixSize.z);

        auto idx = 0;
        int series_counter = series_counter_initial;
        size_t acq_count = 0;
        bool csmSent = false;

        bool weightsEstimated = false;
        auto maxZencode = header.encoding.front().encodingLimits.kspace_encoding_step_2.get().maximum + 1;
        auto maxAcq = ((header.encoding.at(0).encodingLimits.kspace_encoding_step_1.get().maximum + 1) *
                       (header.encoding[0].encodingLimits.kspace_encoding_step_2.get().maximum + 1) *
                       (header.encoding[0].encodingLimits.average.get().maximum + 1) *
                       (header.encoding[0].encodingLimits.repetition.get().maximum +
                        1)); // use -1 for data acquired b/w 12/23 - 01/21

         // Hack for lungwater sequence
         if (header.encoding.at(0).encodingLimits.kspace_encoding_step_2.get().maximum + 1 > 60000)
        {
            maxAcq = maxAcq / (header.encoding.at(0).encodingLimits.kspace_encoding_step_2.get().maximum + 1) * header.encoding.front().encodedSpace.matrixSize.z;
            maxZencode = header.encoding.front().encodedSpace.matrixSize.z;
        }
        GDEBUG_STREAM("maxAcq:" << maxAcq);
        GDEBUG_STREAM("kspace_encoding_step_1:"
                      << header.encoding.front().encodingLimits.kspace_encoding_step_1.get().maximum + 1);
        GDEBUG_STREAM(
            "kspace_encoding_step_2:" << header.encoding.at(0).encodingLimits.kspace_encoding_step_2.get().maximum + 1);
        GDEBUG_STREAM("average:" << (header.encoding[0].encodingLimits.average.get().maximum + 1));
        GDEBUG_STREAM("repetition:" << (header.encoding[0].encodingLimits.repetition.get().maximum + 1));

        std::vector<std::vector<size_t>> idx_phases;
        std::vector<std::vector<std::vector<std::vector<size_t>>>> idx_phases_vec;
        std::vector<Core::Acquisition> allAcq(maxAcq);
        std::vector<ISMRMRD::AcquisitionHeader> headers(maxAcq);
        boost::shared_ptr<nhlbi_toolbox::reconstruction::noncartesian_reconstruction<3>> reconstruction;
        // Collect all the data -- BEGIN()
        bool recon_params_received = false;
        for (auto message : in) {
            if (holds_alternative<Gadgetron::reconParams>(message)) {
                if (recon_params_received == false) {
                    GDEBUG_STREAM("Recon Params received for average reconstruction");
                    recon_params = Core::get<Gadgetron::reconParams>(message);
                    recon_params_received = true;
                } else {
                    GDEBUG_STREAM("Recon Params received for advanced reconstruction");
                    recon_params_adv = Core::get<Gadgetron::reconParams>(message);
                }
            }

            if (holds_alternative<Core::Acquisition>(message) && idx < maxAcq) {
                auto& [head, data, traj] = Core::get<Core::Acquisition>(message);
                allAcq[idx] = std::move(Core::get<Core::Acquisition>(message));

                if ((idx >= int((estimateCSM_perc / 100.0) * maxAcq)) && (!csm_calculated_ && recon_params_received)) {
                    GadgetronTimer timer_CSM("Calculating CSM");
                    GadgetronTimer timer_Average("Calculating Average Image");
                    cudaSetDevice(recon_params.selectedDevices[0]);
                    auto [csm_ims, recon] = estimate_csm(allAcq, idx, recon_params);
                    reconstruction =boost::make_shared<nhlbi_toolbox::reconstruction::noncartesian_reconstruction<3>>(recon);
                    cudaSetDevice(recon_params.selectedDevices[0]);

                    if (reconstruction->get_recon_params().use_gcc)
                    {
                        GadgetronTimer timer("GCC Compression");
                        channel_images =
                            boost::make_shared<cuNDArray<float_complext>>(reconstruction->estimate_gcc_matrix(csm_ims));
                        csm = reconstruction->generateRoemerCSM(&(*channel_images));

                    } else {
                        channel_images = boost::make_shared<cuNDArray<float_complext>>(csm_ims);
                        csm = reconstruction->generateRoemerCSM(&(*channel_images));
                    }
                    timer_CSM.stop();
                    csm_calculated_ = true;
                    if (save_avg) {
                        *channel_images *= *conj(csm.get());
                        auto combined = sum(channel_images.get(), channel_images->get_number_of_dimensions() - 1);
                        cuNDArray<float_complext> cuimages_all =
                            reconstruction->crop_to_recondims<float_complext>(*combined);
                        process_and_send_images(cuimages_all, acqhdr, out, series_counter, "AVG", recon_params);
                        series_counter++;
                        combined->clear();
                    }
                    channel_images->clear();
                    csm_ims.clear();
                    timer_Average.stop();
                }
                idx++;
            }
            if (holds_alternative<std::vector<std::vector<std::vector<std::vector<size_t>>>>>(message)) {
                idx_phases_vec = Core::get<std::vector<std::vector<std::vector<std::vector<size_t>>>>>(message);
            }
        }

        {
            if (!recon_params_received)
                GERROR("Recon Params not received ! Please add PrereconParams.cpp in you xml \n");


            if (!csm_calculated_){
                GadgetronTimer timer_CSM("Calculating CSM At the end");
                GadgetronTimer timer_Average("Calculating Average Image");
                cudaSetDevice(recon_params.selectedDevices[0]);
                auto [csm_ims, recon] = estimate_csm(allAcq, idx, recon_params);
                reconstruction =boost::make_shared<nhlbi_toolbox::reconstruction::noncartesian_reconstruction<3>>(recon);
                cudaSetDevice(recon_params.selectedDevices[0]);

                if (reconstruction->get_recon_params().use_gcc)
                {
                    GadgetronTimer timer("GCC Compression");
                    channel_images =
                        boost::make_shared<cuNDArray<float_complext>>(reconstruction->estimate_gcc_matrix(csm_ims));
                    csm = reconstruction->generateRoemerCSM(&(*channel_images));

                } else {
                    channel_images = boost::make_shared<cuNDArray<float_complext>>(csm_ims);
                    csm = reconstruction->generateRoemerCSM(&(*channel_images));
                }
                timer_CSM.stop();
                csm_calculated_ = true;
                if (save_avg) {
                    *channel_images *= *conj(csm.get());
                    auto combined = sum(channel_images.get(), channel_images->get_number_of_dimensions() - 1);
                    cuNDArray<float_complext> cuimages_all =
                        reconstruction->crop_to_recondims<float_complext>(*combined);
                    process_and_send_images(cuimages_all, acqhdr, out, series_counter, "AVG", recon_params);
                    series_counter++;
                    combined->clear();
                }
                channel_images->clear();
                csm_ims.clear();
                timer_Average.stop();
            }
                

            GadgetronTimer timer("Optimized Recon :");

            // Check if binning data was sent -- cannot proceed without it really ! Use different Gadget
            if (idx_phases_vec.empty() || idx_phases_vec[0].size() == 0)
                GERROR("binning was not done \n");

            // Get bin counts
            auto NE = idx_phases_vec.size();
            auto NC = idx_phases_vec[0].size();
            auto NR = idx_phases_vec[0][0].size();

            GDEBUG_STREAM("Idx_phases: " << "NE: " << NE << " NC: " << NC << " NR: " << NR);

            auto output =
                nhlbi_toolbox::utils::sort_idx_phases(idx_phases_vec, binning_order, binning_collapse_to_last);
            idx_phases = std::get<0>(output);
            shots_per_time = std::get<1>(output);

            // For debugging
            /*
            GDEBUG_STREAM("Numrespbins: " << shots_per_time.get_size(0));
            GDEBUG_STREAM("Numcardbins: " << shots_per_time.get_size(1));
            for (size_t it = 0; it < shots_per_time.get_number_of_elements(); it++){
                GDEBUG_STREAM("it " << it << "SHOTs " <<*(shots_per_time.begin() + it));
                size_t size_phase =(idx_phases[it].size());
                GDEBUG_STREAM("it " << it << "Phase size " << size_phase);
            }
            */

            //////     Recon starting        ///////
            allAcq.resize(idx);

            cudaSetDevice(recon_params.selectedDevices[0]);
            auto& [headAcq, dataAcq, trajAcq] = allAcq[0];
            acqhdr = headAcq;
            RO = dataAcq.get_size(0);
            CHA = dataAcq.get_size(1);
            E2 = this->header.encoding.front().encodedSpace.matrixSize.z;
            N = dataAcq.get_size(3);
            S = 1;
            SLC = 1;
            recon_params.numberChannels = CHA;
            recon_params.RO = RO;
            this->initialize_encoding_space_limits(this->header); // useful ?

            cuNDArray<float_complext> cuData;
            cuNDArray<vector_td<float, 3>>traj;
            cuNDArray<float> dcw;
            std::vector<size_t> number_elements;

            nhlbi_toolbox::reconstruction::noncartesian_reconstruction_3D reconstruction3D(recon_params_adv);
            nhlbi_toolbox::reconstruction::noncartesian_reconstruction_4D reconstruction4D(recon_params_adv);
            nhlbi_toolbox::reconstruction::noncartesian_reconstruction_5D reconstruction5D(recon_params_adv);

            std::vector<cuNDArray<floatd3>> trajVec;
            std::vector<cuNDArray<float>> dcw_estVec;
            std::vector<cuNDArray<float>> dcwVec;

            cuNDArray<Gadgetron::float_complext> cuIimages;
            hoNDArray<std::complex<float>> images_all;

            std::string img_parameters_name = std::string("rType_") + std::to_string(reconType) + std::string("_ite_") +
                                              std::to_string(recon_params.iterations) + std::string("_ls_") +
                                              std::to_string(recon_params.lambda_spatial) + std::string("_lt") +
                                              std::to_string(recon_params.lambda_time);
            if (doConcomitantFieldCorrection) {
                img_parameters_name.append(std::string("_CoCor"));
            }

            switch (reconType) {
                case 4: {
                    GDEBUG_STREAM("Not using default reconstruction pipeline");
                } break;
            default: {
                std::tie(cuData, traj, dcw, number_elements) = reconstruction->organize_data(&allAcq, idx_phases);
                recon_params.shots_per_time = shots_per_time;
                recon_params_adv.shots_per_time = shots_per_time;
                reconstruction->set_recon_params(recon_params);
                reconstruction3D.set_recon_params(recon_params_adv);
                reconstruction4D.set_recon_params(recon_params_adv);
                reconstruction5D.set_recon_params(recon_params_adv);

                trajVec = reconstruction4D.arraytovector(&traj, number_elements);
                // dcw_estVec = reconstruction->arraytovector(&dcw, number_elements);
                if (calculateKPRECOND) {
                    GDEBUG_STREAM("Estimation of DCF using kprecond");
                    dcwVec = reconstruction4D.estimate_kspace_precond_vector(&trajVec);
                } else {
                    GDEBUG_STREAM("Estimation of DCF using GT");
                    dcwVec = reconstruction4D.estimate_dcf(&trajVec);
                }
                
                ////// estimating concomitant field correction weights
                if (doConcomitantFieldCorrection) {
                    nhlbi_toolbox::corrections::mri_concomitant_field_correction_lit field_correction(
                        this->header); // Only do it if concomitant field correction is requested
                    for (auto ii = 0; ii < trajVec.size(); ii++) {
                        auto hoTraj = hoNDArray<floatd3>(
                            std::move(*boost::reinterpret_pointer_cast<hoNDArray<floatd3>>(trajVec[ii].to_host())));
                        std::vector<size_t> non_flat_dims = {recon_params.RO,
                                                            hoTraj.get_number_of_elements() / recon_params.RO};
                        auto traj_view = hoNDArray<floatd3>(non_flat_dims, hoTraj.get_data_ptr());
                        auto gradients = nhlbi_toolbox::utils::traj2grad(traj_view, kspace_scaling, headAcq);
                        if (!weightsEstimated) {
                            field_correction.setup(gradients, headAcq);

                            auto cw = hoNDArray<float_complext>(field_correction.combinationWeights);
                            cweights = boost::make_shared<cuNDArray<float_complext>>(cuNDArray<float_complext>(cw));
                            // nhlbi_toolbox::utils::write_cpu_nd_array(cw,"/opt/data/gt_data/cw_TV.complex");

                            auto x = field_correction.demodulation_freqs;
                            freq_bins = arma::fvec(x.n_elem);
                            for (auto ii = 0; ii < x.n_elem; ii++) {
                                // GDEBUG_STREAM("Freq Elem:" << float(x[ii]) );

                                freq_bins[ii] = float(x[ii]);
                            }
                            weightsEstimated = true;
                        }

                        field_correction.calculate_scaledTime(gradients);
                        auto sct = cuNDArray<float>(field_correction.scaled_time);
                        sctVector.push_back(sct);
                    }
                }


                if (recon_params.use_gcc) {
                auto mtx_vec = reconstruction->get_mtx_vec();
                cuData = reconstruction4D.apply_coil_compression(&cuData, &trajVec, &dcwVec, &mtx_vec);
                }

                for (auto d = 0; d < recon_params.shots_per_time.get_number_of_dimensions(); d++) {
                img_parameters_name.append(std::string("_nbin") + std::to_string(d) + std::string("_") +
                                           std::to_string(recon_params.shots_per_time.get_size(d)));
                }
                
            }
            }

            switch (reconType) {
            case 0: {

                // reconstruction4D.set_recon_params(recon_params);
                if (doConcomitantFieldCorrection)
                    cuIimages = reconstruction4D.reconstructiMOCO_fc(&cuData, &trajVec, &dcwVec, csm, cweights.get(),
                                                                     sctVector, freq_bins, referencePhase);
                else
                    cuIimages = reconstruction4D.reconstructiMOCO(&cuData, &trajVec, &dcwVec, csm, referencePhase);

                process_and_send_images(cuIimages, acqhdr, out, series_counter,
                                        std::string("iMOCO") + img_parameters_name, recon_params);
            } break;

            case 1: {
                if (doConcomitantFieldCorrection)
                    cuIimages = reconstruction4D.reconstruct_fc(&cuData, &trajVec, &dcwVec, csm, cweights.get(),
                                                                sctVector, freq_bins);
                else {
                    cuIimages = reconstruction4D.reconstruct(&cuData, &trajVec, &dcwVec, csm);
                }
                process_and_send_images(cuIimages, acqhdr, out, series_counter,
                                        std::string("4DTresolved") + img_parameters_name, recon_params);
            } break;

            case 2: {
                cuIimages = reconstruction5D.reconstruct(&cuData, &trajVec, &dcwVec, csm);
                // cuIimages.reshape(cuIimages.get_size(0), cuIimages.get_size(1), cuIimages.get_size(2), -1);
                process_and_send_images(cuIimages, acqhdr, out, series_counter,
                                        std::string("5DTresolved") + img_parameters_name, recon_params);
            } break;

            case 3: {
                cuIimages = reconstruction5D.reconstructiMOCO(&cuData, &trajVec, &dcwVec, csm, referencePhase);
                process_and_send_images(cuIimages, acqhdr, out, series_counter,
                                        std::string("4DImoco") + img_parameters_name, recon_params);
            } break;

            case 4: {
                std::vector<size_t> binning_order_respi = {binning_order[1], binning_order[2], binning_order[0]};
                auto output_collapsed = nhlbi_toolbox::utils::sort_idx_phases(idx_phases_vec, binning_order_respi, true,start_idx_nc);
                std::vector<std::vector<size_t>> idx_phases_respiratory = std::get<0>(output_collapsed);
                hoNDArray<size_t> shots_per_time_respiratory = std::get<1>(output_collapsed);
                recon_params.shots_per_time = shots_per_time_respiratory;
                recon_params_adv.shots_per_time = shots_per_time_respiratory;
                reconstruction4D.set_recon_params(recon_params_adv);
                reconstruction->set_recon_params(recon_params);

                GDEBUG_STREAM("Collapsed Numrespbins: " << shots_per_time_respiratory.get_size(0));
                GDEBUG_STREAM("Collapsed Numcardbins: " << shots_per_time_respiratory.get_size(1));
                /*
                for (size_t it = 0; it < shots_per_time_respiratory.get_number_of_elements(); it++){
                    GDEBUG_STREAM("it " << it << "SHOTs " <<*(shots_per_time_respiratory.begin() + it));
                    size_t size_phase =(idx_phases_respiratory[it].size());
                    GDEBUG_STREAM("it " << it << "Phase size " << size_phase);
                }
                */
                auto [cuData_respi, traj_respi, dcw_respi, number_elements_respi] =reconstruction->organize_data(&allAcq, idx_phases_respiratory);
                std::vector<cuNDArray<floatd3>> trajVec_respi =reconstruction4D.arraytovector(&traj_respi, number_elements_respi);
                std::vector<cuNDArray<float>> dcwVec_respi = reconstruction4D.estimate_dcf(&trajVec_respi);
                auto ave_cuIimages = reconstruction4D.reconstruct(&cuData_respi, &trajVec_respi, &dcwVec_respi, csm,false);
                process_and_send_images(ave_cuIimages, acqhdr, out, series_counter,
                                        std::string("4DTresolved") + img_parameters_name, recon_params);
                series_counter++;
                cuData_respi.clear();
                trajVec_respi.clear();
                dcwVec_respi.clear();
                auto [cuData_All, traj_rc, dcw_rc, number_elements_rc] =reconstruction->organize_data(&allAcq, idx_phases);
                recon_params.shots_per_time = shots_per_time;
                recon_params_adv.shots_per_time = shots_per_time;
                reconstruction5D.set_recon_params(recon_params_adv);
                reconstruction->set_recon_params(recon_params);

                for (size_t it = 0; it < shots_per_time.get_number_of_elements(); it++) {
                    GDEBUG_STREAM("it " << it << "SHOTs " << *(shots_per_time.begin() + it));
                    size_t size_phase = (idx_phases[it].size());
                    GDEBUG_STREAM("it " << it << "Phase size " << size_phase);
                }
                std::vector<cuNDArray<floatd3>> trajVec_respi_cardiac =reconstruction5D.arraytovector(&traj_rc, number_elements_rc);
                std::vector<cuNDArray<float>> dcwVec_respi_cardiac =reconstruction5D.estimate_dcf(&trajVec_respi_cardiac);
                cuIimages = reconstruction5D.reconstructiMOCO_avg_image(&cuData_All, &trajVec_respi_cardiac, &dcwVec_respi_cardiac, ave_cuIimages, csm, referencePhase);
                process_and_send_images(cuIimages, acqhdr, out, series_counter,std::string("4DImoco") + img_parameters_name, recon_params);
            } break;

            case 5: {
                cuIimages = reconstruction4D.reconstructLR(&cuData, &trajVec, &dcwVec, csm);
                process_and_send_images(cuIimages, acqhdr, out, series_counter,
                                        std::string("4DLR") + img_parameters_name, recon_params);
            } break;

            case 6: {
                cuIimages = reconstruction4D.reconstructMOCOLR(&cuData, &trajVec, &dcwVec, csm, referencePhase);
                // Process and send the respiratory compensated images
                //auto cuImages_compensated = reconstruction4D.register_and_apply_deformations(cuIimages,referencePhase);
                process_and_send_images(cuIimages, acqhdr, out, series_counter,
                                        std::string("4D_MOCOLR") + img_parameters_name, recon_params);
                //series_counter++;
                //process_and_send_images(cuImages_compensated, acqhdr, out, series_counter,
                //                        std::string("MOCOLR_comp") + img_parameters_name, recon_params);
                
                
                
            } break;

            case 7: // 3D recon
            {
                // Loop over each set of indices in idx_phases for 3D recon
                auto counter = 0;
                auto cuImages_all =
                    cuNDArray<float_complext>(reconstruction3D.image_dims_[0], reconstruction3D.image_dims_[1],
                                              recon_params.rmatrixSize.z, idx_phases.size());
                auto im_dims = *cuImages_all.get_dimensions();
                auto stride = std::accumulate(im_dims.begin(), im_dims.end() - 1, 1, std::multiplies<size_t>());
                im_dims.pop_back(); // Remove the last dimension (number of phases)
                for (const auto& idx : idx_phases) {
                    GDEBUG_STREAM("Processed 3D recon for indices: " << counter);
                    auto [cuData3D, traj3D, dcw3D, number_elements3D] = reconstruction->organize_data(&allAcq, {idx});
                    dcw3D = reconstruction->estimate_dcf(&traj3D, &dcw3D);
                    // auto cuIimages = cuNDArray<float_complext>(im_dims, cuImages_all.data() + stride * counter);
                    auto cuIimages = reconstruction3D.reconstruct_CGSense(&cuData3D, &traj3D, &dcw3D, csm);
                    cudaMemcpy(cuImages_all.get_data_ptr() + stride * counter, cuIimages.get_data_ptr(),
                               stride * sizeof(float_complext), cudaMemcpyDefault);
                    counter++;
                }
                series_counter = 1;
                images_all = hoNDArray<std::complex<float>>(std::move(
                    *boost::reinterpret_pointer_cast<hoNDArray<std::complex<float>>>(cuImages_all.to_host())));
                process_and_send_images(cuImages_all, acqhdr, out, series_counter,
                                        std::string("3DSense") + img_parameters_name, recon_params);
            } break;
            case 8: {
                std::vector<size_t> binning_order_respi = {binning_order[1], binning_order[2], binning_order[0]};
                auto output_collapsed = nhlbi_toolbox::utils::sort_idx_phases(idx_phases_vec, binning_order_respi, true,start_idx_nc);
                std::vector<std::vector<size_t>> idx_phases_respiratory = std::get<0>(output_collapsed);
                hoNDArray<size_t> shots_per_time_respiratory = std::get<1>(output_collapsed);
                recon_params.shots_per_time = shots_per_time_respiratory;
                recon_params_adv.shots_per_time = shots_per_time_respiratory;
                reconstruction4D.set_recon_params(recon_params_adv);
                reconstruction->set_recon_params(recon_params);

                GDEBUG_STREAM("Collapsed Numrespbins: " << shots_per_time_respiratory.get_size(0));
                GDEBUG_STREAM("Collapsed Numcardbins: " << shots_per_time_respiratory.get_size(1));

                auto [cuData_respi, traj_respi, dcw_respi, number_elements_respi] =reconstruction->organize_data(&allAcq, idx_phases_respiratory);
                std::vector<cuNDArray<floatd3>> trajVec_respi =reconstruction4D.arraytovector(&traj_respi, number_elements_respi);
                std::vector<cuNDArray<float>> dcwVec_respi = reconstruction4D.estimate_dcf(&trajVec_respi);
                auto ave_cuIimages = reconstruction4D.reconstructMOCOLR(&cuData_respi, &trajVec_respi, &dcwVec_respi, csm);
                process_and_send_images(ave_cuIimages, acqhdr, out, series_counter,
                                        std::string("4DMOCOLR") + img_parameters_name, recon_params);
                series_counter++;
                cuData_respi.clear();
                trajVec_respi.clear();
                dcwVec_respi.clear();
                auto [cuData_All, traj_rc, dcw_rc, number_elements_rc] =reconstruction->organize_data(&allAcq, idx_phases);
                recon_params.shots_per_time = shots_per_time;
                recon_params_adv.shots_per_time = shots_per_time;
                reconstruction5D.set_recon_params(recon_params_adv);
                reconstruction->set_recon_params(recon_params);

                for (size_t it = 0; it < shots_per_time.get_number_of_elements(); it++) {
                    GDEBUG_STREAM("it " << it << "SHOTs " << *(shots_per_time.begin() + it));
                    size_t size_phase = (idx_phases[it].size());
                    GDEBUG_STREAM("it " << it << "Phase size " << size_phase);
                }
                std::vector<cuNDArray<floatd3>> trajVec_respi_cardiac =reconstruction5D.arraytovector(&traj_rc, number_elements_rc);
                std::vector<cuNDArray<float>> dcwVec_respi_cardiac =reconstruction5D.estimate_dcf(&trajVec_respi_cardiac);
                cuIimages = reconstruction5D.reconstructiMOCO_avg_image(&cuData_All, &trajVec_respi_cardiac, &dcwVec_respi_cardiac, ave_cuIimages, csm, referencePhase);
                process_and_send_images(cuIimages, acqhdr, out, series_counter,std::string("4DImoco") + img_parameters_name, recon_params);
            } break;
            default:
                GDEBUG_STREAM("Invalid reconType");
            }
            using namespace Gadgetron::Indexing;

            series_counter++;

        }
    }

    void process_and_send_images_ho_float(hoNDArray<float> &images, ISMRMRD::AcquisitionHeader &acqhdr, OutputChannel &out, int &series_counter, char *image_comment)
    {
        hoNDArray<std::complex<float>> images_cmplx(*images.get_dimensions()); 
        real_to_complex<std::complex<float>>(images,images_cmplx);
        process_and_send_images_ho(images_cmplx, acqhdr, out, series_counter, image_comment);

    }
    void process_and_send_images_ho(hoNDArray<std::complex<float>> &images, ISMRMRD::AcquisitionHeader &acqhdr, OutputChannel &out, int &series_counter, char *image_comment)
    {
        IsmrmrdImageArray imarray_sense;

        // nhlbi_toolbox::utils::filterImagealongSlice(images_all, get_kspace_filter_type(ftype), fwidth, fsigma);
        // nhlbi_toolbox::utils::filterImage(images_all, get_kspace_filter_type(inftype), infwidth, infsigma);

        auto tmp = hoNDArray<std::complex<float>>(images);
        imarray_sense.data_ = tmp;

        nhlbi_toolbox::utils::attachHeadertoImageArray(imarray_sense, acqhdr, this->header);

        prepare_image_array(imarray_sense, (size_t)0, ((int)series_counter), GADGETRON_IMAGE_REGULAR);
        imarray_sense.meta_[0].append(GADGETRON_IMAGECOMMENT, image_comment);
        imarray_sense.meta_[0].append(GADGETRON_SEQUENCEDESCRIPTION, image_comment);

        out.push(imarray_sense);
        imarray_sense.meta_.clear();
        imarray_sense.headers_.clear();
    }

    void process_and_send_images(cuNDArray<float_complext>& cuImages, ISMRMRD::AcquisitionHeader& acqhdr,
                                 OutputChannel& out, int& series_counter, std::string image_comment,
                                 Gadgetron::reconParams& recon_params) {
        size_t NDim = cuImages.get_number_of_dimensions();
        size_t CHA = 1;
        size_t N = NDim > 3 ? cuImages.get_size(3) : 1;
        size_t S = NDim > 4 ? cuImages.get_size(4) : 1;
        size_t SLC = NDim > 5 ? cuImages.get_size(5) : 1;
        GDEBUG_STREAM("CuImage SIZE " << NDim << " [RO E1 E2 CHA N S SLC] = [" << cuImages.get_size(0) << " "
                                      << cuImages.get_size(1) << " " << cuImages.get_size(2) << " " << CHA << " " << N
                                      << " " << S << " " << SLC << "] ");
        IsmrmrdImageArray imarray_sense;

        auto images = hoNDArray<std::complex<float>>(
            std::move(*boost::reinterpret_pointer_cast<hoNDArray<std::complex<float>>>(cuImages.to_host())));
        auto tmp = hoNDArray<std::complex<float>>(images);
        tmp.reshape(tmp.get_size(0), tmp.get_size(1), tmp.get_size(2), 1, N, S, SLC);
        imarray_sense.data_ = tmp;
        nhlbi_toolbox::utils::attachHeadertoImageArrayReconParams(imarray_sense, acqhdr, recon_params);
        prepare_image_array(imarray_sense, (size_t)0, ((int)series_counter), GADGETRON_IMAGE_REGULAR);
        GDEBUG_STREAM("Final IMAGE INDEX " << imarray_sense.headers_(N - 1, S - 1, SLC - 1).image_index);
        GDEBUG_STREAM("Image Name :" << image_comment)
        for (uint16_t loc = 0; loc < SLC; loc++) {
            for (uint16_t s = 0; s < S; s++) {
                for (uint16_t n = 0; n < N; n++) {

                    size_t offset = n + s * N + loc * N * S;
                    imarray_sense.headers_(n, s, loc).image_index = offset + 1;
                    imarray_sense.meta_[offset].append(GADGETRON_IMAGECOMMENT, image_comment.c_str());
                    imarray_sense.meta_[offset].append(GADGETRON_SEQUENCEDESCRIPTION, image_comment.c_str());
                }
            }
        }
        out.push(imarray_sense);
        imarray_sense.meta_.clear();
        imarray_sense.headers_.clear();
    }
    std::tuple<cuNDArray<float_complext>, nhlbi_toolbox::reconstruction::noncartesian_reconstruction<3>>
    estimate_csm(std::vector<Core::Acquisition> acqs, size_t idxs, Gadgetron::reconParams recon_params) {
        GDEBUG_STREAM("idxs size: " << idxs);

        acqs.resize(idxs);
        cudaSetDevice(recon_params.selectedDevices[0]);

        auto& [headAcq, dataAcq, trajAcq] = acqs[0];
        auto acqhdr = headAcq;
        auto RO = dataAcq.get_size(0);
        auto CHA = dataAcq.get_size(1);
        auto E2 = this->header.encoding.front().encodedSpace.matrixSize.z;
        auto N = dataAcq.get_size(3);
        auto S = 1;
        auto SLC = 1;

        recon_params.numberChannels = CHA;
        recon_params.RO = RO;

        this->initialize_encoding_space_limits(this->header);

        nhlbi_toolbox::reconstruction::noncartesian_reconstruction reconstruction(recon_params);

        auto [cuData, traj, dcf_csm] = reconstruction.organize_data(&acqs);
        GDEBUG_STREAM("acqs size: " << acqs.size());
        GDEBUG_STREAM("traj size: " << traj.get_size(0));
        if (calculateKPRECOND) {
            GDEBUG_STREAM("Estimation of DCF using kprecond");
            dcf_csm = reconstruction.estimate_kspace_precond(&traj);
        } else {
            GDEBUG_STREAM("Estimation of DCF using GT");
            dcf_csm = reconstruction.estimate_dcf(&traj, &dcf_csm);
        }

        square_inplace(&dcf_csm);
        cuNDArray<float_complext> channel_images(reconstruction.get_recon_dims());
        {
            GadgetronTimer timer("Reconstruct CSM");

            reconstruction.reconstruct(&cuData, &channel_images, &traj, &dcf_csm);
        }
        // if (recon_params.use_gcc)
        // {
        //     GadgetronTimer timer("GCC Compression");
        //     channel_images = reconstruction.estimate_gcc_matrix(channel_images);
        // }
        (cuData.clear());
        (traj.clear());
        (dcf_csm.clear());
        // auto csm = reconstruction.generateRoemerCSM(&channel_images);

        return std::make_tuple(channel_images, reconstruction);
    }

  protected:
    NODE_PROPERTY(Debug, double, "Debug", 0);
    NODE_PROPERTY(doConcomitantFieldCorrection, bool, "doConcomitantFieldCorrection", true);
    NODE_PROPERTY(referencePhase, float, "referencePhase between 0 and 1", 0.5);
    NODE_PROPERTY(estimateCSM_perc, float, "perc_data_csm", 50);
    NODE_PROPERTY(reconType, size_t, "Recontype 0:iMOCO, 1:4DTresolved, 2: 5DTresolved, 3: 4DImoco,7: 3Dall", 0);
    NODE_PROPERTY(sliding_window, float, "sliding_window for lungwater", 0.33);
    NODE_PROPERTY(window_duration, float, "window_duration for dlungwater", 1.5);

    NODE_PROPERTY(binning_order, std::vector<size_t>, "Binning Order (Nr,Nc,Ne)", (std::vector<size_t>{0, 1, 2}));
    NODE_PROPERTY(binning_collapse_to_last, bool, "Collapse binning to last dimension", false);
    NODE_PROPERTY(start_idx_nc, float, "With Collapse binning, only subsample NC", 0);
    NODE_PROPERTY(series_counter_initial, int, "series_counter_initial", 0);
    NODE_PROPERTY(save_avg, bool, "Saving Average image", true);
    NODE_PROPERTY(calculateKPRECOND, bool, "GT DCF of Kspace preconditioning", false);
};

GADGETRON_GADGET_EXPORT(Noncart_recon_gadget)
