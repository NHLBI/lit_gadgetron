#include "noncartesian_reconstruction.h"
#include <cuNonCartesianSenseOperator.h>
#include <cuSbcCgSolver.h>

using namespace Gadgetron;
using namespace Gadgetron::Core;
using namespace Gadgetron::Indexing;

namespace nhlbi_toolbox
{
    namespace reconstruction
    {
        class noncartesian_reconstruction_2Dt : public noncartesian_reconstruction<2>
        {
        public:
            noncartesian_reconstruction_2Dt(reconParams recon_params) : noncartesian_reconstruction<2>(recon_params){};
            
            using noncartesian_reconstruction::reconstruct;
            using noncartesian_reconstruction::organize_data;
            //using noncartesian_reconstruction::organize_data_vector;
            //using noncartesian_reconstruction::organize_data_hoNDArray;             

            cuNDArray<float_complext> reconstruct_CGSense(cuNDArray<float_complext> *data,
            std::vector<cuNDArray<vector_td<float, 2>>> *traj,std::vector<cuNDArray<float>> *dcw,
            boost::shared_ptr<cuNDArray<float_complext>> csm);

            cuNDArray<float_complext> reconstruct_CGSense_wav(cuNDArray<float_complext> *data,
            std::vector<cuNDArray<vector_td<float, 2>>> *traj,std::vector<cuNDArray<float>> *dcw,
            boost::shared_ptr<cuNDArray<float_complext>> csm);
            
            std::tuple<cuNDArray<float_complext>,
            std::vector<cuNDArray<vector_td<float, 2>>>,
            std::vector<cuNDArray<float>>>
            organize_data(
                hoNDArray<float_complext> *data,
                hoNDArray<vector_td<float, 2>> *traj,
                hoNDArray<float> *dcw);
            
    
        };
    }
}