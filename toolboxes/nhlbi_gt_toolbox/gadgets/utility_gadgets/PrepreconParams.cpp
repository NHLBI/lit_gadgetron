#include <Node.h>
#include <mri_core_utility.h>
#include <mri_core_acquisition_bucket.h>
#include <ismrmrd/xml.h>
#include <gadgetron_mricore_export.h>
#include <mri_core_def.h>
#include <mri_core_data.h>
#include <hoNDArray_math.h>
#include <hoNDArray_utils.h>
#include <hoNDArray.h>
#include <ismrmrd/ismrmrd.h>
#include <GadgetronTimer.h>
#include "util_functions.h"
#include "reconParams.h"

using namespace Gadgetron;
using namespace Gadgetron::Core;

/*
     Need to be tested on 2D , stack of spiral, stack of radial images
*/

class PrepreconParams : public ChannelGadget<Core::Acquisition>
{

public:
    PrepreconParams(const Core::Context &context, const Core::GadgetProperties &props) : ChannelGadget<Acquisition>(context, props)
    {

    }
    void process(InputChannel<Acquisition> &in, OutputChannel &out) override
    {   
        bool downsampling_flag = std::find(downsampling_vector.begin(), downsampling_vector.end(), 1) == downsampling_vector.end();
        auto downsampling_plane=downsampling_vector[0];
        auto downsampling_z=downsampling_vector[1];
        bool acq_ref_flag =false;

        Gadgetron::reconParams recon_params;
        Gadgetron::reconParams recon_params_avg;

        std::vector<int> selectedGPUs;
        auto eligibleGPUs = nhlbi_toolbox::utils::FindCudaDevices(0);
        
        std::istringstream iss(selectedDevices_STR); // Create an input string stream from the string
        std::string token;
        std::vector<float> selectedDevices;
        GDEBUG_STREAM("DEVICE SELECTION: " << selectedDevices_STR);
        // Extract tokens separated by whitespace
        while (iss >> token) { 
            selectedDevices.push_back(std::stof(token));
        }

        GDEBUG_STREAM("DEVICE SELECTION: " << selectedDevices[0] << " " << selectedDevices[1]);
        for (auto idx_d=0; idx_d <selectedDevices.size(); idx_d++){

            if (selectedDevices[idx_d]>=0){
                selectedGPUs.push_back(int(selectedDevices[idx_d]));
            }
            if (selectedDevices[idx_d]==-1) {
                std::vector<int> GPUs_not_yet_selected;
                std::set_difference(eligibleGPUs.begin(), eligibleGPUs.end(), selectedGPUs.begin(),selectedGPUs.end(), std::back_inserter(GPUs_not_yet_selected));
                if (!GPUs_not_yet_selected.empty()){
                    selectedGPUs.push_back(GPUs_not_yet_selected[0]);
                }
            }
        }
        if (selectedGPUs.empty()) {
            GERROR("No GPU selected for reconstruction, please check the configuration.");
            return;
        }
        auto repeated_GPUs=1; // Ongoing issue with multiple repeated GPUs
        std::vector<int> total_selected_GPUs(selectedGPUs.size() * repeated_GPUs);
        for (std::size_t rep = 0; rep < repeated_GPUs; ++rep) {
            std::copy(selectedGPUs.begin(), selectedGPUs.end(), std::next(total_selected_GPUs.begin(), rep * selectedGPUs.size()));
        }

        for (auto idx_d=0; idx_d <total_selected_GPUs.size(); idx_d++){
            GDEBUG_STREAM("Chosen DEVICE: " << total_selected_GPUs[idx_d]);
        }
        unsigned int warp_size = cudaDeviceManager::Instance()->warp_size();
        std::vector<unsigned int> warp_vector = {1,1,1};
        for (auto iwarp = 0; iwarp < warpCUDA_vector.size(); iwarp++){
            if (warpCUDA_vector[iwarp]){
                warp_vector[iwarp]=warp_size;
            }

        }


        recon_params.lambda_spatial = lambda_spatial;
        recon_params.lambda_spatial_imoco = lambda_spatial_imoco; //not used
        recon_params.lambda_time = lambda_time;
        recon_params.lambda_time2 = lambda_time2;
        recon_params.lambda_LR = lambda_LR;
        recon_params.block_LR = block_LR;
        recon_params.iterations = iterations;
        recon_params.iterations_imoco = iterations_imoco;
        recon_params.iterations_inner = iterations_inner;
        recon_params.iteration_count_moco = iteration_count_moco;
        recon_params.doMC_iter = doMC_iter;
        recon_params.tolerance = tolerance;
        recon_params.norm = norm;
        recon_params.maxIteRegistration = maxIteRegistration;

        //NUFFT parameters
        recon_params.oversampling_factor_ = oversampling_factor;
        recon_params.kernel_width_ = kernel_width;
        
        //DCF parameters
        recon_params.useIterativeDCWEstimated = useIterativeDCWEstimated;
        recon_params.iterations_dcf = iterations_dcf;
        recon_params.oversampling_factor_dcf_ = oversampling_factor_dcf; 
        recon_params.kernel_width_dcf_ = kernel_width_dcf; 

        //GCC parameters
        recon_params.use_gcc = use_gcc;
        recon_params.gcc_coils = gcc_coils;

        recon_params.selectedDevice = selectedGPUs[0];
        recon_params.selectedDevices = selectedGPUs;

        recon_params.try_channel_gridding=try_channel_gridding;



        recon_params.ematrixSize = this->header.encoding.front().encodedSpace.matrixSize;
        recon_params.rmatrixSize = this->header.encoding.front().reconSpace.matrixSize;
        
        GDEBUG_STREAM("Raw Header information :");
        GDEBUG_STREAM("Encoded Matrix: X " << header.encoding.front().encodedSpace.matrixSize.x << " Y " << header.encoding.front().encodedSpace.matrixSize.y << " Z " << header.encoding.front().encodedSpace.matrixSize.z);
        GDEBUG_STREAM("Recon Matrix: X " << header.encoding.front().reconSpace.matrixSize.x << " Y " << header.encoding.front().reconSpace.matrixSize.y << " Z " << header.encoding.front().reconSpace.matrixSize.z);
        GDEBUG_STREAM("Encoded FOV: X " << header.encoding.front().encodedSpace.fieldOfView_mm.x << " Y " << header.encoding.front().encodedSpace.fieldOfView_mm.y << " Z " << header.encoding.front().encodedSpace.fieldOfView_mm.z);
        GDEBUG_STREAM("Parameters: FOV OVERSAMPLING  x " << matOSP_vector[0] << " Y " << matOSP_vector[1] << " Z " << matOSP_vector[2] << " Downsampling Plane " << downsampling_plane << " Z " << downsampling_z <<  " Warp VECTOR X " << warp_vector[0] << " Y " << warp_vector[1] << " Z " << warp_vector[2] );
        

        if (recon_params.ematrixSize.z ==1 && recon_params.rmatrixSize.z >1){
            GERROR("MRD Header broken ! Required to wait the end of the acquisition before doing anything ");
        }


        // Downsampling matrix
        auto me_dx=float(recon_params.ematrixSize.x)/downsampling_plane;
        auto me_dy=float(recon_params.ematrixSize.y)/downsampling_plane;
        auto me_dz=float(recon_params.ematrixSize.z)/downsampling_z;

        GDEBUG_STREAM("me factor: X " << me_dx  << " Y " << me_dy << " Z " << me_dz );
        
        recon_params.omatrixSize.x = size_t(me_dx);
        recon_params.omatrixSize.y = size_t(me_dy);
        recon_params.omatrixSize.z = size_t(me_dz);

        // Large FOV matrix + CUDA warping
        auto me_x=size_t(ceil((matOSP_vector[0]*me_dx)/warp_vector[0]))*warp_vector[0];
        auto me_y=size_t(ceil((matOSP_vector[1]*me_dy)/warp_vector[1]))*warp_vector[1];
        auto me_z=size_t(ceil((matOSP_vector[2]*me_dz)/warp_vector[2]))*warp_vector[2];

        auto mr_dx=float(recon_params.rmatrixSize.x)/downsampling_plane;
        auto mr_dy=float(recon_params.rmatrixSize.y)/downsampling_plane;
        auto mr_dz=float(recon_params.rmatrixSize.z)/downsampling_z;

        auto mr_x=size_t(ceil((matOSP_vector[0]*mr_dx)/warp_vector[0]))*warp_vector[0];
        auto mr_y=size_t(ceil((matOSP_vector[1]*mr_dy)/warp_vector[1]))*warp_vector[1];
        auto mr_z=size_t(ceil((matOSP_vector[2]*mr_dz)/warp_vector[2]))*warp_vector[2];

        if (recon_params.ematrixSize.z ==1 && recon_params.rmatrixSize.z ==1){
            auto mr_z=1;
            auto me_z=1;
        }

        recon_params.ematrixSize.x = me_x;
        recon_params.ematrixSize.y = me_y;
        recon_params.ematrixSize.z = me_z;

        recon_params.rmatrixSize.x = mr_x;
        recon_params.rmatrixSize.y = mr_y;
        recon_params.rmatrixSize.z = mr_z;
        
        recon_params.fov = this->header.encoding.front().encodedSpace.fieldOfView_mm;
        recon_params.fov.x=recon_params.fov.x*(mr_x/mr_dx);
        recon_params.fov.y=recon_params.fov.y*(mr_y/mr_dy);
        recon_params.fov.z=recon_params.fov.z*(mr_z/mr_dz);

        GDEBUG_STREAM("Encoded Matrix: X " << recon_params.ematrixSize.x  << " Y " << recon_params.ematrixSize.y << " Z " << recon_params.ematrixSize.z );
        GDEBUG_STREAM("Recon Matrix: X " << recon_params.rmatrixSize.x << " Y " << recon_params.rmatrixSize.y << " Z " << recon_params.rmatrixSize.z);
        GDEBUG_STREAM("Recon FOV: X " << recon_params.fov.x << " Y " << recon_params.fov.y << " Z " << recon_params.fov.z);

        
        // recon params avg for specifying specific dcf parameters

        //Matrix

        recon_params_avg.ematrixSize.x =recon_params.ematrixSize.x;
        recon_params_avg.ematrixSize.y =recon_params.ematrixSize.y;
        recon_params_avg.ematrixSize.z =recon_params.ematrixSize.z;

        recon_params_avg.rmatrixSize.x =recon_params.rmatrixSize.x;
        recon_params_avg.rmatrixSize.y =recon_params.rmatrixSize.y;
        recon_params_avg.rmatrixSize.z =recon_params.rmatrixSize.z;

        recon_params_avg.omatrixSize.x =recon_params.omatrixSize.x;
        recon_params_avg.omatrixSize.y =recon_params.omatrixSize.y;
        recon_params_avg.omatrixSize.z =recon_params.omatrixSize.z;

        //FOV
        recon_params_avg.fov.x =recon_params.fov.x;
        recon_params_avg.fov.y =recon_params.fov.y;
        recon_params_avg.fov.z =recon_params.fov.z;

        GDEBUG_STREAM("AVERAGE RECON PARAMS" )
        GDEBUG_STREAM("Encoded Matrix: X " << recon_params_avg.ematrixSize.x  << " Y " << recon_params_avg.ematrixSize.y << " Z " << recon_params_avg.ematrixSize.z );
        GDEBUG_STREAM("Recon Matrix: X " << recon_params_avg.rmatrixSize.x << " Y " << recon_params_avg.rmatrixSize.y << " Z " << recon_params_avg.rmatrixSize.z);
        GDEBUG_STREAM("Recon FOV: X " << recon_params_avg.fov.x << " Y " << recon_params_avg.fov.y << " Z " << recon_params_avg.fov.z);

        

        //NUFFT parameters
        recon_params_avg.oversampling_factor_ = oversampling_factor;
        recon_params_avg.kernel_width_ = kernel_width;
        
        //DCF parameters
        recon_params_avg.useIterativeDCWEstimated = useIterativeDCWEstimated_avg;
        recon_params_avg.iterations_dcf = iterations_dcf_avg;
        recon_params_avg.oversampling_factor_dcf_ = oversampling_factor_dcf_avg; 
        recon_params_avg.kernel_width_dcf_ = kernel_width_dcf_avg; 

        recon_params_avg.use_gcc = use_gcc;
        recon_params_avg.gcc_coils = gcc_coils;

        recon_params_avg.selectedDevice = selectedGPUs[0];
        recon_params_avg.selectedDevices = selectedGPUs;

        recon_params_avg.try_channel_gridding=try_channel_gridding;
        GDEBUG_STREAM("CHANNEL GRIDDING " << recon_params_avg.try_channel_gridding << " " << recon_params.try_channel_gridding);
        size_t RO = 0;
        uint16_t crop_index_st = 0;
        uint16_t crop_index_end = 0;
        for (auto message : in)
        {

            using namespace Gadgetron::Indexing;
            auto &[head, data, traj] = message;

            if ((head.isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_NAVIGATION_DATA) || head.isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_HPFEEDBACK_DATA) || head.isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_RTFEEDBACK_DATA)))
            {
                out.push(Core::Acquisition(std::move(head), std::move(data), std::move(traj)));
                continue;
            }
            if (!downsampling_flag){
                if(RO==0){
                    
                    RO = data.get_size(0);
                    auto CHA = data.get_size(1);
                    GDEBUG_STREAM("RO " << RO);
                    GDEBUG_STREAM("CHA " << CHA);

                    recon_params.numberChannels = CHA;
                    recon_params.RO = RO;
                    recon_params_avg.numberChannels = CHA;
                    recon_params_avg.RO = RO;
                    out.push(recon_params_avg);
                    out.push(recon_params);
                }

                out.push(Core::Acquisition(std::move(head), std::move(data), std::move(traj)));
                continue;
            }
            else{
                hoNDArray<std::complex<float>> data_short;
                hoNDArray<float> low_res_trajectory_and_weights;
                auto trajectory_and_weights = *traj;
                
                if (header.encoding.front().encodedSpace.matrixSize.z > 1){
                    auto traj_dcw = nhlbi_toolbox::utils::separate_traj_and_dcw_2or3<3>(&trajectory_and_weights);
                    auto traj_sep = std::move(*std::get<0>(traj_dcw).get());
                    auto dcw_sep = std::move(*std::get<1>(traj_dcw).get());
                    if (RO == 0)
                    { 
                        float r_min=0.0;
                        for (size_t i = 0; i < dcw_sep.get_number_of_elements(); i++){
                            if (is3D){
                                r_min= std::sqrt(traj_sep[i][0]*traj_sep[i][0]+traj_sep[i][1]*traj_sep[i][1]+traj_sep[i][2]*traj_sep[i][2]);
                            }
                            else{
                                r_min= std::sqrt(traj_sep[i][0]*traj_sep[i][0]+traj_sep[i][1]*traj_sep[i][1]);
                            }
                            if (r_min <= 0.5f/downsampling_plane){
                                crop_index_end = i;
                            }
                            if (r_min > 0.5f/downsampling_plane and crop_index_end==0){
                                crop_index_st = (i+1);
                            }
                        }
                        
                        RO=(crop_index_end-crop_index_st)+1;
                        auto CHA = data.get_size(1);
                        GDEBUG_STREAM("crop_index_end " << crop_index_end);
                        GDEBUG_STREAM("crop_index_st " << crop_index_st);
                        GDEBUG_STREAM("RO " << RO);
                        GDEBUG_STREAM("CHA " << CHA);
                       
                        recon_params.numberChannels = CHA;
                        recon_params.RO = RO;
                        recon_params_avg.numberChannels = CHA;
                        recon_params_avg.RO = RO;
                        out.push(recon_params_avg);
                        out.push(recon_params);
                    }
                    
                    low_res_trajectory_and_weights.create({4,RO});
                    for (size_t ii = 0; ii < RO; ii++){
                        low_res_trajectory_and_weights(0, ii) = (traj_sep[ii+crop_index_st][0]*downsampling_plane > 0.5f) ? 0.5f : ((traj_sep[ii+crop_index_st][0]*downsampling_plane < -0.5f) ? -0.5f : traj_sep[ii+crop_index_st][0]*downsampling_plane);
                        low_res_trajectory_and_weights(1, ii) = (traj_sep[ii+crop_index_st][1]*downsampling_plane > 0.5f) ? 0.5f : ((traj_sep[ii+crop_index_st][1]*downsampling_plane < -0.5f) ? -0.5f : traj_sep[ii+crop_index_st][1]*downsampling_plane);
                        low_res_trajectory_and_weights(2, ii) = (traj_sep[ii+crop_index_st][2]*downsampling_z > 0.5f) ? 0.5f : ((traj_sep[ii+crop_index_st][2]*downsampling_z < -0.5f) ? -0.5f : traj_sep[ii+crop_index_st][2]*downsampling_z);
                        low_res_trajectory_and_weights(3, ii) = dcw_sep(ii+crop_index_st);
                    }
                    data_short.create({RO, head.active_channels});
                    #pragma omp parallel
                    #pragma omp for
                    for (int ii = 0; ii < RO; ii++)
                    {
                        data_short(ii, slice) = data(ii+crop_index_st, slice);
                    }
                    head.number_of_samples = data_short.get_size(0);
                }
                else{
                    auto traj_dcw = nhlbi_toolbox::utils::separate_traj_and_dcw_2or3<2>(&trajectory_and_weights);
                    auto traj_sep = std::move(*std::get<0>(traj_dcw).get());
                    auto dcw_sep = std::move(*std::get<1>(traj_dcw).get());
                    if (RO == 0)
                    { 
                        float r_min=0.0;
                        for (size_t i = 0; i < dcw_sep.get_number_of_elements(); i++){
                            r_min= sqrt(traj_sep[i][0]*traj_sep[i][0]+traj_sep[i][1]*traj_sep[i][1]);

                            if (r_min <= 0.5f/downsampling_plane){
                                crop_index_end = i;
                            }
                            if (r_min > 0.5f/downsampling_plane and crop_index_end==0){
                                crop_index_st = (i+1);
                            }
                        }
                        RO=(crop_index_end-crop_index_st)+1;
                        auto CHA = data.get_size(1);
                        GDEBUG_STREAM("RO " << RO);
                        GDEBUG_STREAM("CHA " << CHA);

                        recon_params.numberChannels = CHA;
                        recon_params.RO = RO;
                        recon_params_avg.numberChannels = CHA;
                        recon_params_avg.RO = RO;
                        out.push(recon_params_avg);
                        out.push(recon_params);
                    }
                    
                    low_res_trajectory_and_weights.create({4,RO});
                    for (size_t ii = 0; ii < RO; ii++){
                        low_res_trajectory_and_weights(0, ii) = (traj_sep[ii+crop_index_st][0]*downsampling_plane > 0.5f) ? 0.5f : ((traj_sep[ii+crop_index_st][0]*downsampling_plane < -0.5f) ? -0.5f : traj_sep[ii+crop_index_st][0]*downsampling_plane);
                        low_res_trajectory_and_weights(1, ii) = (traj_sep[ii+crop_index_st][1]*downsampling_plane > 0.5f) ? 0.5f : ((traj_sep[ii+crop_index_st][1]*downsampling_plane < -0.5f) ? -0.5f : traj_sep[ii+crop_index_st][1]*downsampling_plane);
                        low_res_trajectory_and_weights(2, ii) = dcw_sep(ii+crop_index_st);
                    }
                    data_short.create({RO, head.active_channels});
                    #pragma omp parallel
                    #pragma omp for
                    for (int ii = 0; ii < RO; ii++)
                    {
                        data_short(ii, slice) = data(ii+crop_index_st, slice);
                    }
                    head.number_of_samples = data_short.get_size(0);
                }
                out.push(Core::Acquisition(std::move(head), std::move(data_short), std::move(low_res_trajectory_and_weights)));
            }           
        }


        
    }

protected:
    NODE_PROPERTY(matOSP_vector, std::vector<float>, "Large FOV factor",(std::vector<float>{ 1, 1, 1})); // Vector of scaling factors for large Field Of View (FOV) 
    NODE_PROPERTY(downsampling_vector, std::vector<float>, "Downsampling factor plane(x,y) and z)",(std::vector<float>{ 1, 1})); // Downsampling factors for plane (x, y) and z dimension
    NODE_PROPERTY(warpCUDA_vector, std::vector<bool>, "Warp CUDA (32)",(std::vector<bool>{ true, true, false})); // Flags for respecting CUDA  size of warp ( matrix x,y,z should be a multiple of 32)

    NODE_PROPERTY(is3D, bool, "is 3D not stack of 2D", false); // Flag indicating if data is 3D non cartesian (not a stack of stars, spirals)
    //NUFFT parameters
    NODE_PROPERTY(kernel_width, float, "kernel_width NUFFT", 3); // NUFFT kernel width parameter
    NODE_PROPERTY(oversampling_factor, float, "oversampling_factor", 1.5); // NUFFT oversampling factor
    //DCF parameters
    NODE_PROPERTY(kernel_width_dcf, float, "kernel_width_dcf", 5.5); // DCF kernel width parameter
    NODE_PROPERTY(iterations_dcf, size_t, "iterations_dcf", 10); // Number of iterations for DCF estimation
    NODE_PROPERTY(oversampling_factor_dcf, float, "oversampling_factor_dcf", 2.1); // DCF oversampling factor
    NODE_PROPERTY(useIterativeDCWEstimated, bool, "Iterative DCW with Estimates", false); // Flag to use iterative DCF estimation

    NODE_PROPERTY(kernel_width_dcf_avg, float, "kernel_width_dcf", 5.5); // DCF kernel width parameter for average recon
    NODE_PROPERTY(iterations_dcf_avg, size_t, "iterations_dcf", 10); // Number of iterations for DCF estimation for average recon
    NODE_PROPERTY(oversampling_factor_dcf_avg, float, "oversampling_factor_dcf", 2.1); // DCF oversampling factor for average recon
    NODE_PROPERTY(useIterativeDCWEstimated_avg, bool, "Iterative DCW with Estimates", false); // Flag to use iterative DCF estimation for average recon

    //Regularization parameters
    NODE_PROPERTY(lambda_spatial, float, "lambda spatial", 0); // Lambda for spatial regularization
    NODE_PROPERTY(lambda_spatial_imoco, float, "lambda spatial imoco", 1e-1); // Lambda for spatial regularization in motion compensation
    NODE_PROPERTY(lambda_time, float, "lambda time", 0); // Lambda for temporal regularization
    NODE_PROPERTY(lambda_time2, float, "lambda time 2", 0); // Lambda for second temporal regularization
    NODE_PROPERTY(lambda_LR, float, "lambda for LR LLR MOCOLR", 0); // Lambda for low-rank regularization
    NODE_PROPERTY(block_LR, float, "block size for LLR", 32); // Block size for low-rank regularization (Not used currently,fully implemented)
    
    NODE_PROPERTY(iterations, size_t, "Number of Iterations", 1); // Number of iterations for reconstruction
    NODE_PROPERTY(iterations_imoco, size_t, "Number of Iterations imoco", 5); // Number of iterations for motion-compensated reconstruction
    NODE_PROPERTY(iterations_inner, size_t, "Number of Iterations inner", 2); // Number of inner iterations for reconstruction
    NODE_PROPERTY(iteration_count_moco, size_t, "iteration_count_moco", 3); // Number of motion compensation iterations
    NODE_PROPERTY(doMC_iter, bool, "doMC_iter", false); // Flag to calculate motion fields at each iteration

    NODE_PROPERTY(tolerance, float, "tolerance", 1e-3); // Tolerance for convergence
    NODE_PROPERTY(norm, size_t, "Norm", 2); // Norm type for regularization (1: L1, 2: L2)
    // GCC parameters (Not used currently,fully implemented)
    NODE_PROPERTY(use_gcc, bool, "use_gcc", false); // Flag to use GCC calibration
    NODE_PROPERTY(gcc_coils, size_t, "gcc_coils", 6); // Number of coils for GCC calibration
    NODE_PROPERTY(selectedDevices_STR, std::string, "String list of GPU device (0-N:device i, -1 : let GT choose, -2: No Device)", "-1 -2"); // String for selecting GPU devices
    //NODE_PROPERTY(repeated_GPUs, unsigned int, "Repeat eligible GPUs x times",1);
    NODE_PROPERTY(maxIteRegistration, int, "Number of Iterations with estimation registration", 0); // Number of iterations for registration with estimation
    NODE_PROPERTY(try_channel_gridding, bool, "try_gridding over all channels", true); // Flag to enable  gridding over all channels
    
    };

GADGETRON_GADGET_EXPORT(PrepreconParams)