#pragma once

#include <vector_td.h>
#include <vector>
#include <complex>
#include <hoNDArray.h>
#include <hoNDFFT.h>
#include <hoNDArray_utils.h>
#include <hoNDArray_math.h>
#include <boost/optional.hpp>
#include <hoNDArray_fileio.h>
#include <boost/math/constants/constants.hpp>
#include <math.h>
#include <stdio.h>
#include <boost/algorithm/string.hpp>
#include <iostream>
#include <string>
#include <fstream>
#include <boost/filesystem/fstream.hpp>
#include <hoArmadillo.h>
#include <cuSDC.h>
#include <cuNDArray_math.h>
#include <cuNDArray.h>
#include <cuNDFFT.h>
#include <mri_core_utility.h> // Added MRD
#include "cuNonCartesianTSenseOperator.h"
#include <cuCgPreconditioner.h>
#include <cuPartialDerivativeOperator2.h>
#include <cuPartialDerivativeOperator.h>
#include <cuNDArray_utils.h>
#include "densityCompensation.h"

#include "sense_utilities.h"
#include "cuGpBbSolver.h"
#include <cuSbcCgSolver.h>
#include <ismrmrd/ismrmrd.h>

#include "util_functions.h"

#include <python_toolbox.h>
#include <filesystem>
#include "reconParams.h"
#include <cuNDFFT.h>

#include "cuNFFT.h"
#include <vector_td_utilities.h>
#include "complext.h"
//#include "real_utilities.h"
#include "GriddingConvolution.h"

using namespace Gadgetron;
using namespace Gadgetron::Core;
using namespace Gadgetron::Indexing;

namespace nhlbi_toolbox
{
    namespace reconstruction
    {
        template <size_t D = 3> class noncartesian_reconstruction
        {
        public:


            noncartesian_reconstruction(reconParams recon_params);

            void reconstruct(
                cuNDArray<float_complext> *data,
                cuNDArray<float_complext> *image,
                cuNDArray<vector_td<float, D>> *traj,
                cuNDArray<float> *dcw);

            void reconstruct(
                cuNDArray<float_complext> *data,
                cuNDArray<float_complext> *image,
                cuNDArray<vector_td<float, D>> *traj,
                cuNDArray<float> *dcw,
                cuNDArray<float_complext> *csm);

            void deconstruct(
                cuNDArray<float_complext> *images,
                cuNDArray<float_complext> *data,
                cuNDArray<vector_td<float, D>> *traj,
                cuNDArray<float> *dcw,
                cuNDArray<float_complext> *csm);

            void deconstruct(
                cuNDArray<float_complext> *images,
                cuNDArray<float_complext> *data,
                cuNDArray<vector_td<float, D>> *traj,
                cuNDArray<float> *dcw);

            boost::shared_ptr<cuNDArray<float_complext>> generateCSM(
                cuNDArray<float_complext> *channel_images);

            boost::shared_ptr<cuNDArray<float_complext>> generateMcKenzieCSM(
                cuNDArray<float_complext> *channel_images);

            boost::shared_ptr<cuNDArray<float_complext>> generateRoemerCSM(
                cuNDArray<float_complext> *channel_images);


            // This is the standard method takes in data, traj, and dcw loads things to GPU for recon (reconstruction.cpp)
            std::tuple<cuNDArray<float_complext>,
                       cuNDArray<vector_td<float, D>>,
                       cuNDArray<float>> organize_data_hoNDArray(
                hoNDArray<float_complext> *data,
                hoNDArray<vector_td<float, D>> *traj,
                hoNDArray<float> *dcw,
                bool calculateDCF=true,
                bool calculateKPRECOND=false);
            
            std::tuple<cuNDArray<float_complext>,
                        std::vector<cuNDArray<vector_td<float, D>>>,
                        std::vector<cuNDArray<float>>>
            organize_data_vector(
                hoNDArray<float_complext> *data,
                hoNDArray<vector_td<float, D>> *traj,
                hoNDArray<float> *dcw,
                bool calculateDCF=true,
                bool calculateKPRECOND=false);
            
            // This is the standard method takes in data, traj, and dcw loads things to GPU for recon (deconstruction.cpp)
            std::tuple<cuNDArray<vector_td<float, D>>,
                       cuNDArray<float>> organize_traj_dcw_hoNDArray(
                hoNDArray<vector_td<float, D>> *traj,
                hoNDArray<float> *dcw,
                bool calculateDCF=true,
                bool calculateKPRECOND=false);

            boost::shared_ptr<cuNDArray<float_complext>> generateEspiritCSM(cuNDArray<float_complext> *channel_images);
            cuNDArray<float_complext> estimate_gcc_matrix(cuNDArray<float_complext> images);
            cuNDArray<float_complext> apply_gcc_matrix(cuNDArray<float_complext> images, std::vector<hoNDArray<std::complex<float>>> mtx);
            
            // This is the optimized method takes in acq and returns data,traj and dcw for recon - skips accumulate gadget 
            std::tuple<cuNDArray<float_complext>,
                       cuNDArray<vector_td<float, D>>,
                       cuNDArray<float>>
            organize_data(std::vector<Core::Acquisition> *allAcq);

            std::tuple<cuNDArray<float_complext>,
                       cuNDArray<vector_td<float, D>>,
                       cuNDArray<float>,
                       std::vector<size_t>>
            organize_data(std::vector<Core::Acquisition> *allAcq, std::vector<std::vector<size_t>> idx_phases);

            std::vector<size_t> get_recon_dims() { return recon_dims; };

            template <typename T>
            cuNDArray<T> crop_to_recondims(cuNDArray<T> &input);

            boost::shared_ptr<cuNFFT_plan<float, D>> nfft_plan_;
            std::vector<size_t> image_dims_;
            std::vector<size_t> image_dims_os_;

            cuNDArray<float> estimate_dcf(cuNDArray<vector_td<float, 3>> *traj);
            
            
            cuNDArray<float> estimate_dcf(cuNDArray<vector_td<float, 3>> *traj,std::vector<size_t> number_elements);
            std::vector<cuNDArray<float>> estimate_dcf(std::vector<cuNDArray<vector_td<float, 3>>> *traj);

            cuNDArray<float> estimate_dcf(cuNDArray<vector_td<float, 2>> *traj);
            cuNDArray<float> estimate_dcf(cuNDArray<vector_td<float, 2>> *traj,std::vector<size_t> number_elements);
            std::vector<cuNDArray<float>> estimate_dcf(std::vector<cuNDArray<vector_td<float, 2>>> *traj);
            cuNDArray<float> estimate_dcf(cuNDArray<vector_td<float, 2>> *traj,cuNDArray<float> *dcf_in);
            cuNDArray<float> estimate_dcf(cuNDArray<vector_td<float, 3>> *traj,cuNDArray<float> *dcf_in);
            template <typename T > std::vector<cuNDArray<T>> arraytovector(cuNDArray<T> *inputArray, std::vector<size_t> number_elements);
            template <typename T > std::vector<cuNDArray<T>> arraytovector(cuNDArray<T> *inputArray, hoNDArray<size_t> number_elements);
            template <typename T > cuNDArray<T>  vectortoarray(std::vector<cuNDArray<T>> *inputArray);
            std::vector<cuNDArray<float>> estimate_dcf(std::vector<cuNDArray<vector_td<float, 3>>> *traj, std::vector<cuNDArray<float>> *dcf_in);
            cuNDArray<float> estimate_kspace_precond(cuNDArray<vector_td<float, D>> *traj);
            std::vector<cuNDArray<float>> estimate_kspace_precond_vector(std::vector<cuNDArray<vector_td<float, D>>> *traj);
            
            cuNDArray<float> estimate_dcf_special(cuNDArray<vector_td<float, 3>> *traj);
       
            reconParams get_recon_params() const {
                return recon_params;
            }

            void set_recon_params(reconParams new_recon_params) {
                recon_params = new_recon_params;
            }

            void apply_gcc_compress(cuNDArray<float_complext> &images, cuNDArray<float_complext> mtx, size_t dim);
            
            void CC(cuNDArray<float_complext> &DATA, cuNDArray<float_complext> mtx, size_t dim);
            
            std::vector<cuNDArray<float_complext>> get_mtx_vec(){
                return mtx_vec;
            }

            void set_mtx_vec(std::vector<cuNDArray<float_complext>> new_mtx_vec) {
                mtx_vec = new_mtx_vec;
            }
            
        protected:
            reconParams recon_params;
            float resx;
            float resy;
            float resz;
            std::vector<size_t> recon_dims;
            std::vector<size_t> recon_dims_encodSpace;
            std::vector<size_t> recon_dims_reconSpace;
            density_compensation dcfO;
            std::vector<cuNDArray<float_complext>> mtx_vec;
            bool isprocessed = false;
            std::filesystem::path python_path ;

        private:
        };
    }
}