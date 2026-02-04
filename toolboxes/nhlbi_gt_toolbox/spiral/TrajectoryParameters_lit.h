#pragma once
 
#include <ismrmrd/xml.h>
#include <log.h>
#include <Gadget.h>
#include "vds.h"
#include <hoNDArray_fileio.h>
#include <vector_td_utilities.h>
#include <mri_core_utility.h>
#include <mri_core_girf_correction.h>
#include <hoArmadillo.h>
#include <hoNDArray_utils.h>
#include <hoNDArray_math.h>
#include <hoNDArray_utils.h>
#include <hoNDArray_elemwise.h>
using namespace Gadgetron;
 
namespace Gadgetron
{
    namespace Spiral
    {
 
        class TrajectoryParameters_lit
        {
        public:
            TrajectoryParameters_lit() = default;
            TrajectoryParameters_lit(const ISMRMRD::IsmrmrdHeader &h);
            double get_Tsampling_us();
            std::pair<hoNDArray<floatd2>, hoNDArray<float>>
            calculate_trajectories_and_weight(const ISMRMRD::AcquisitionHeader &acq_header);
            void set_girf_sampling_time(float time);
            void set_acceleration_factor(size_t acc);

            void read_girf_kernel(std::string girf_folder);
            hoNDArray<std::complex<float>> get_girf_kernel();
 
            // NHLBI customisation
            // 2-FOV variable density design
            double  vds_factor_;
            // custom rotation number
            long    spiral_rotations_;
 
        private:
            Core::optional<hoNDArray<std::complex<float>>> girf_kernel;
            float girf_sampling_time_us;
            long Tsamp_ns_;
            long Nints_;
            double gmax_;
            double smax_;
            double krmax_;
            double fov_;
            float TE_;
            size_t acc;
            std::string systemModel;
 
            hoNDArray<floatd2> correct_gradients(const hoNDArray<floatd2> &gradients, float grad_samp_us,
                                                 float girf_samp_us, const float *read_dir, const float *phase_dir,
                                                 const float *slice_dir);
            
 
        };
    } // namespace Spiral
} // namespace Gadgetron