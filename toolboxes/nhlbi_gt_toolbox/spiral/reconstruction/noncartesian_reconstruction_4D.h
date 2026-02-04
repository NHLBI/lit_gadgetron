#pragma once
#include "noncartesian_reconstruction.h"
#include "cuNonCartesianTSenseOperator_fc.h"
#include "cuNonCartesianMOCOOperator_fc.h"
#include "cuNonCartesianMOCOOperator.h"

#include <cuSbcCgSolver.h>
#include <util_functions.h>
#include <cuNlcgSolver.h>

using namespace Gadgetron;
using namespace Gadgetron::Core;
using namespace Gadgetron::Indexing;

namespace nhlbi_toolbox
{
    namespace reconstruction
    {
        class noncartesian_reconstruction_4D : public noncartesian_reconstruction<3>
        {
        public:
            noncartesian_reconstruction_4D(reconParams recon_params) : noncartesian_reconstruction<3>(recon_params) {};

            using noncartesian_reconstruction::organize_data;
            using noncartesian_reconstruction::organize_data_hoNDArray;
            using noncartesian_reconstruction::organize_data_vector;
            using noncartesian_reconstruction::reconstruct;
            
            cuNDArray<float> registration_images(cuNDArray<float_complext> *images_all);

            cuNDArray<float_complext> reconstruct(
                cuNDArray<float_complext> *data,
                std::vector<cuNDArray<floatd3>> *traj,
                std::vector<cuNDArray<float>> *dcw,
                boost::shared_ptr<cuNDArray<float_complext>> csm,
                bool crop_image=true);

            cuNDArray<float_complext> reconstruct(
                cuNDArray<float_complext> *data,
                std::vector<cuNDArray<floatd3>> *traj,
                std::vector<cuNDArray<float>> *dcw,
                boost::shared_ptr<cuNDArray<float_complext>> csm,
                cuNDArray<float_complext> *x0);

            cuNDArray<float_complext> reconstruct_nlcg(
                cuNDArray<float_complext> *data,
                std::vector<cuNDArray<floatd3>> *traj,
                std::vector<cuNDArray<float>> *dcw,
                boost::shared_ptr<cuNDArray<float_complext>> csm);

            cuNDArray<float_complext> reconstruct_fc(
                cuNDArray<float_complext> *data,
                std::vector<cuNDArray<floatd3>> *traj,
                std::vector<cuNDArray<float>> *dcw,
                boost::shared_ptr<cuNDArray<float_complext>> csm,
                cuNDArray<float_complext> *combination_weights,
                std::vector<cuNDArray<float>> &scaled_time,
                arma::fvec fbins);

            cuNDArray<float_complext> reconstructiMOCO_fc(
                cuNDArray<float_complext> *data,
                std::vector<cuNDArray<floatd3>> *traj,
                std::vector<cuNDArray<float>> *dcw,
                boost::shared_ptr<cuNDArray<float_complext>> csm,
                cuNDArray<float_complext> *combination_weights,
                std::vector<cuNDArray<float>> &scaled_time,
                arma::fvec fbins,
                float referencePhase=0.49);

            cuNDArray<float_complext> reconstructLR(
            cuNDArray<float_complext> *data,
            std::vector<cuNDArray<floatd3>> *traj,
            std::vector<cuNDArray<float>> *dcw,
            boost::shared_ptr<cuNDArray<float_complext>> csm);

            cuNDArray<float_complext> reconstructLLR(
            cuNDArray<float_complext> *data,
            std::vector<cuNDArray<floatd3>> *traj,
            std::vector<cuNDArray<float>> *dcw,
            boost::shared_ptr<cuNDArray<float_complext>> csm);

            cuNDArray<float_complext> reconstructMOCOLR(
            cuNDArray<float_complext> *data,
            std::vector<cuNDArray<floatd3>> *traj,
            std::vector<cuNDArray<float>> *dcw,
            boost::shared_ptr<cuNDArray<float_complext>> csm,
            float referencePhase=0.49);

            cuNDArray<float_complext> reconstructiMOCO(
                cuNDArray<float_complext> *data,
                std::vector<cuNDArray<floatd3>> *traj,
                std::vector<cuNDArray<float>> *dcw,
                boost::shared_ptr<cuNDArray<float_complext>> csm,
                float referencePhase=0.49);

                cuNDArray<float_complext> apply_coil_compression(
                cuNDArray<float_complext> *data,
                std::vector<cuNDArray<floatd3>> *traj,
                std::vector<cuNDArray<float>> *dcw,
                std::vector<cuNDArray<float_complext>> *mtx);

            std::tuple<cuNDArray<float_complext>,
                       std::vector<cuNDArray<floatd3>>,
                       std::vector<cuNDArray<float>>>
            organize_data(
                hoNDArray<float_complext> *data,
                hoNDArray<floatd3> *traj,
                hoNDArray<float> *dcw,
                bool calculateDCF = true,
                bool calculateKPRECOND =false);

            std::tuple<cuNDArray<float_complext>,
                       std::vector<cuNDArray<floatd3>>,
                       std::vector<cuNDArray<float>>>
            organize_data(
                std::vector<hoNDArray<float_complext>> *data,
                std::vector<hoNDArray<floatd3>> *traj,
                std::vector<hoNDArray<float>> *dcw);

            std::tuple<std::vector<cuNDArray<floatd3>>,
                       std::vector<cuNDArray<float>>>
            organize_data(
                cuNDArray<floatd3> *traj,
                cuNDArray<float> *dcw,
                std::vector<size_t> number_elements);

            std::tuple<cuNDArray<float>, cuNDArray<float>> register_images_time(hoNDArray<float> images_all, unsigned int referenceIndex, unsigned int level, std::vector<double> iters, std::vector<double> regularization_hilbert_strength, std::vector<double> LocalCCR_sigmaArg, bool BidirectionalReg, bool DivergenceFreeReg, bool verbose, std::string sim_meas, bool useInvDef);
            std::tuple<std::vector<cuNDArray<float>>, std::vector<cuNDArray<float>>> register_images_gpu(cuNDArray<float_complext> images_all, float referencePhase);
            cuNDArray<float_complext> register_and_apply_deformations(cuNDArray<float_complext> images_all, float referencePhase);
            cuNDArray<float_complext> cu_applyDeformations(cuNDArray<float_complext> cuIimages, std::vector<cuNDArray<float>> deformations);
        
            void set_recon_params(const reconParams& params) {
                this->recon_params = params;
            }
            hoNDArray<std::complex<float>> applyDeformations(hoNDArray<std::complex<float>> images_all, cuNDArray<float> deformation);
        };
    }
}