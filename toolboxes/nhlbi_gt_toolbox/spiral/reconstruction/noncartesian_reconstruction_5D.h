#pragma once
#include "noncartesian_reconstruction_4D.h"
#include "cuNonCartesianTSenseOperator_fc.h"
#include "cuNonCartesianMOCOOperator_fc.h"
#include "cuNonCartesianMOCOOperator.h"

#include <cuSbcCgSolver.h>
//#include <motion_correction.h>
#include <util_functions.h>
#include <cuNlcgSolver.h>

using namespace Gadgetron;
using namespace Gadgetron::Core;
using namespace Gadgetron::Indexing;


namespace nhlbi_toolbox
{
    namespace reconstruction
    {
        class noncartesian_reconstruction_5D : public noncartesian_reconstruction_4D
        {
        public:
            noncartesian_reconstruction_5D(reconParams recon_params) : noncartesian_reconstruction_4D(recon_params){};
            using noncartesian_reconstruction_4D::reconstruct;
            using noncartesian_reconstruction_4D::reconstruct_nlcg;

            cuNDArray<float_complext> reconstruct(
                cuNDArray<float_complext> *data,
                std::vector<cuNDArray<floatd3>> *traj,
                std::vector<cuNDArray<float>> *dcw,
                boost::shared_ptr<cuNDArray<float_complext>> csm,
                bool crop_image = true);

            cuNDArray<float_complext> reconstruct_nlcg(
                cuNDArray<float_complext> *data,
                std::vector<cuNDArray<floatd3>> *traj,
                std::vector<cuNDArray<float>> *dcw,
                boost::shared_ptr<cuNDArray<float_complext>> csm);

            cuNDArray<float_complext> reconstructiMOCO(
                cuNDArray<float_complext> *data,
                std::vector<cuNDArray<floatd3>> *traj,
                std::vector<cuNDArray<float>> *dcw,
                boost::shared_ptr<cuNDArray<float_complext>> csm,
                float referencePhase=0.49);
            
            cuNDArray<float_complext> reconstructiMOCO_withdef(
                cuNDArray<float_complext> *data,
                std::vector<cuNDArray<floatd3>> *traj,
                std::vector<cuNDArray<float>> *dcw,
                boost::shared_ptr<cuNDArray<float_complext>> csm,
                cuNDArray<float> *def,
                cuNDArray<float> *invdef,
                float referencePhase=0.49);


            cuNDArray<float_complext> reconstructiMOCO_avg_image(
                cuNDArray<float_complext> *data,
                std::vector<cuNDArray<floatd3>> *traj,
                std::vector<cuNDArray<float>> *dcw,
                cuNDArray<float_complext> avg_images,
                boost::shared_ptr<cuNDArray<float_complext>> csm,
                float referencePhase=0.49);
        };
    }
}