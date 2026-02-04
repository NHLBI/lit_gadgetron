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
#include "util_functions.h"
#include <mri_core_utility.h> // Added MRD

using namespace Gadgetron;
using namespace Gadgetron::Core;
using namespace Gadgetron::Indexing;

namespace nhlbi_toolbox
{
    namespace reconstruction
    {
        class density_compensation
        {
        public:
            density_compensation() = default;
            std::vector<arma::uvec> extractSlices(hoNDArray<floatd3> sp_traj, std::vector<size_t> image_dims);

            template <unsigned int D>
            cuNDArray<float> estimate_DCF(cuNDArray<vector_td<float, D>> &traj, cuNDArray<float> &dcw, std::vector<size_t> image_dims);
            template <unsigned int D>
            cuNDArray<float> estimate_DCF(cuNDArray<vector_td<float, D>> &traj, std::vector<size_t> image_dims);

            hoNDArray<float> estimate_DCF_slice(hoNDArray<floatd3> &sp_traj, hoNDArray<float> &sp_dcw, std::vector<size_t> image_dims , bool do3D=true);
            hoNDArray<float> estimate_DCF_slice(hoNDArray<floatd3> &sp_traj, std::vector<size_t> image_dims, bool do3D=false);

            hoNDArray<floatd2> traj3Dto2D(hoNDArray<floatd3> &sp_traj);

            bool useIterativeDCWEstimated = false;
            float iterations = 15;
            float oversampling_factor_ = 2.1;
            float kernel_width_ = 5.5;

        private:
            float kspace_scaling;

            std::vector<size_t> image_dims_;
            uint64d3 image_dims_os_;
            bool verbose;
            size_t maxDevices = 8;
        };
    }
}