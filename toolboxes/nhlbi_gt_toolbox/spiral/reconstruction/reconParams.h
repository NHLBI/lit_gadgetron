#pragma once
#include <hoNDArray.h>
#include <ismrmrd/ismrmrd.h>

namespace Gadgetron
{
    struct reconParams
    {
        ISMRMRD::MatrixSize ematrixSize;
        ISMRMRD::MatrixSize rmatrixSize;
        ISMRMRD::MatrixSize omatrixSize;
        ISMRMRD::FieldOfView_mm fov;
        hoNDArray<size_t> shots_per_time;
        size_t numberChannels;
        size_t RO;
        float oversampling_factor_;
        float kernel_width_;
        size_t iterations = 10;
        size_t iterations_imoco = 10;
        size_t iterations_inner = 2;
        float tolerance = 1e-3;
        size_t iterations_dcf = 10;
        float kernel_width_dcf_ = 2.1;
        float oversampling_factor_dcf_ = 2.1;
        int selectedDevice = 0;
        std::vector<int> selectedDevices ;
        float lambda_spatial = 1e-1;
        float lambda_spatial_imoco = 1e-1;
        float lambda_time = 1e-1;
        float lambda_time2 = 1e-1;
        float lambda_LR = 1e-1;
        size_t block_LR = 32;
        size_t norm = 2;
        bool useIterativeDCWEstimated = false;
        bool use_gcc = false;
        size_t gcc_coils = 10;
        bool doMC_iter = false;
        int maxIteRegistration = 10; 
        size_t iteration_count_moco = 3;
        bool try_channel_gridding = true;
    };
}