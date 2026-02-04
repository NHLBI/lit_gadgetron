
/**
 * @file cuNonCartesianMOCOOperator_fc.cpp
 * @brief GPU-accelerated Non-Cartesian Motion-Corrected Reconstruction Operator
 *
 * This file implements a CUDA-based operator for non-Cartesian MRI reconstruction
 * with motion correction capabilities. The operator supports:
 *
 * - Multi-GPU parallel processing
 * - Motion correction using deformation fields
 * - Field correction for concomitant gradients
 * - Both forward (image->k-space) and adjoint (k-space->image) operations
 * - Flexible threading for optimal GPU utilization
 *
 * Key Features:
 * - Supports single or multiple GPU devices
 * - Thread-safe device context management
 * - Efficient memory management across devices
 * - Integration with motion correction algorithms
 *
 * @author NHLBI Gadgetron Toolbox
 */

#include "cuNonCartesianMOCOOperator_fc.h"
#include "gpuRegistration.cuh"
#include "util_functions.h"

using namespace Gadgetron;

/**
 * @brief Forward operation: Cartesian image -> Non-Cartesian k-space data
 *
 * This function performs the forward encoding operation, converting Cartesian
 * image data to non-Cartesian k-space data. Includes motion correction by
 * applying deformation fields to account for subject motion during acquisition.
 *
 * The operation is parallelized across multiple GPU devices and time points,
 * with each iteration handling a specific temporal frame of the reconstruction.
 *
 * @param in Input Cartesian image data
 * @param out Output non-Cartesian k-space data
 * @param accumulate Whether to accumulate results or overwrite output
 */
template <class REAL, unsigned int D>
void cuNonCartesianMOCOOperator_fc<REAL, D>::mult_M(cuNDArray<complext<REAL>> *in, cuNDArray<complext<REAL>> *out,
                                                    bool accumulate)
{
    // ========== INPUT VALIDATION ==========
    // Validate input and output arrays are not null
    if (!in || !out)
    {
        throw std::runtime_error("cuNonCartesianSenseOperator::mult_M : 0x0 input/output not accepted");
    }

    // Validate array dimensions match expected domain and codomain
    if (!in->dimensions_equal(&this->domain_dims_) || !out->dimensions_equal(&this->codomain_dims_))
    {
        throw std::runtime_error(
            "cuNonCartesianSenseOperator::mult_H: input/output arrays do not match specified domain/codomains");
    }

    // ========== DIMENSION SETUP ==========
    // Forward operation: Cartesian image -> Non-Cartesian k-space data
    std::vector<size_t> full_dimensions = *this->get_domain_dimensions();   // Cartesian image dimensions
    std::vector<size_t> data_dimensions = *this->get_codomain_dimensions(); // Non-Cartesian data dimensions

    data_dimensions.pop_back(); // Remove coil dimension from data dimensions

    int dims_orig = full_dimensions.size();

    // Reduce dimensions to spatial dimensions only (remove extra dimensions beyond D)
    for (int ii = 0; ii < (dims_orig - D); ii++)
    {
        full_dimensions.pop_back();
    }

    full_dimensions.push_back(this->ncoils_); // Add coil dimension back

    // ========== DEVICE AND MEMORY SETUP ==========
    int cur_device = in->get_device();
    cudaSetDevice(cur_device);

    // Create input array view and calculate strides for memory access
    std::vector<size_t> slice_dimensions = *this->get_domain_dimensions();
    auto input = cuNDArray<complext<REAL>>(slice_dimensions, in->data());

    // Reduce to spatial dimensions only (X, Y, Z)
    dims_orig = slice_dimensions.size();
    for (int ii = 0; ii < (dims_orig - D); ii++)
        slice_dimensions.pop_back();

    // Calculate stride for accessing different time points in input data
    auto stride = std::accumulate(slice_dimensions.begin(), slice_dimensions.end(), 1,
                                  std::multiplies<size_t>()); // product of X,Y,and Z

    // Calculate stride for output data access
    std::vector<size_t> tmp_dims = *this->get_codomain_dimensions();
    auto stride_data = std::accumulate(tmp_dims.begin(), tmp_dims.end() - 1, 1, std::multiplies<size_t>());

    auto tmpview_dims = full_dimensions;

    // ========== PROCESSING INITIALIZATION ==========
    GadgetronTimer timer("Deconstruct");

    // Initialize motion correction and GPU registration objects
    cuNonCartesianMOCOOperator<float, D> mocoObj(this->convolutionType);
    gpuRegistration gr;

    bool failure = true;

    // ========== PARALLEL PROCESSING LOOP ==========
    // Process each time point/shot in parallel across available GPUs
    // Each iteration handles one temporal frame of the reconstruction
    cuNDArray<complext<REAL>> slice_view_out;
    cuNDArray<complext<REAL>> tmp_out;
    #pragma omp parallel for num_threads(std::min({eligibleGPUs.size(), this->shots_per_time_.get_size(0)})) shared(in, out, scaled_time_, fbins_, combinationWeights_, tmpview_dims, trajectory_, nrfc_vector) private(full_dimensions, slice_view_out,tmp_out) ordered
    for (size_t it = 0; it < this->shots_per_time_.get_number_of_elements(); it++)
    {
        // Set device context and determine which GPU to use for this iteration
        cudaSetDevice(cur_device);
        auto gpuDevice = eligibleGPUs[it % eligibleGPUs.size()]; // Distribute work across available GPUs

        // Calculate accumulated offset for output data placement
        auto inter_acc = std::accumulate(this->shots_per_time_.begin(), this->shots_per_time_.begin() + it, size_t(0));

        // ========== INPUT DATA PREPARATION ==========
        // Prepare dimensions for this iteration's input slice
        full_dimensions = tmpview_dims;
        full_dimensions.pop_back(); // Remove coil dimension temporarily

        // Extract input slice for current time point
        auto slice_input = cuNDArray<complext<REAL>>(full_dimensions, input.data() + int(it / this->shots_per_time_.get_size(0)) * stride);

        full_dimensions.push_back(this->ncoils_); // Add coil dimension back

        // ========== MOTION CORRECTION SETUP ==========
        // Determine which deformation field to use for this iteration
        size_t indexDef = 0;
        if (backward_deformation_.size() == this->shots_per_time_.get_size(0))
            indexDef = it % this->shots_per_time_.get_size(0); // Use respiratory-specific deformation
        else
            indexDef = it; // Use time-specific deformation

        // Apply motion correction by deforming the input image
        auto bdef = nhlbi_toolbox::utils::set_device(&(this->backward_deformation_)[indexDef], cur_device);
        auto slice_view_in = gr.deform_image(&slice_input, bdef);

        // ========== OUTPUT DATA PREPARATION ==========
        // Set up output dimensions for this iteration's k-space data
        auto ddims = data_dimensions;
        ddims.pop_back(); // Remove interleave dimension
        ddims.push_back(*(this->shots_per_time_.begin() + it)); // Add correct interleave count for this iteration
        ddims.push_back(this->ncoils_); // Add coil dimension

        // Create output arrays for this iteration
        slice_view_out.create(ddims);
        
        cuNDArray<complext<REAL>> tmp;

        // Switch to target GPU and initialize reconstruction object
        cudaSetDevice(gpuDevice);
        nhlbi_toolbox::reconstruction::noncartesian_reconstruction_fc nrfc(this->recon_params_);

        // ========== RECONSTRUCTION EXECUTION ==========
        // Choose reconstruction path based on device configuration
        if (cur_device != gpuDevice)
        {
            // Cross-device reconstruction: requires temporary arrays
            if (accumulate)
            {
                GDEBUG_STREAM("accumulate true" << gpuDevice);
                tmp_out.create(ddims);
                nrfc.reconstruct_todevice(tmp_out, slice_view_in, trajectory_[it], this->dcw_[it], *this->csm_.get(), *combinationWeights_, scaled_time_[it], fbins_, gpuDevice, false);
                slice_view_out += tmp_out;
                }
                else
                {
                    // Direct reconstruction to output array
                    nrfc.reconstruct_todevice(slice_view_out, slice_view_in, trajectory_[it], this->dcw_[it], *this->csm_.get(), *combinationWeights_, scaled_time_[it], fbins_, gpuDevice, false);
                }
        }
        else
        {
            // Same-device reconstruction: more efficient, direct operations
            if (accumulate)
            {
                GDEBUG_STREAM("accumulate true" << gpuDevice);
                tmp_out.create(ddims);
                nrfc.deconstruct(slice_view_in, tmp_out, trajectory_[it], this->dcw_[it], *this->csm_.get(), *combinationWeights_, scaled_time_[it], fbins_);
                slice_view_out += tmp_out;
            }
            else
            {
                // Direct deconstruction to output array
                nrfc.deconstruct(slice_view_in, slice_view_out, trajectory_[it], this->dcw_[it], *this->csm_.get(), *combinationWeights_, scaled_time_[it], fbins_);
            }
        }

        // ========== OUTPUT DATA TRANSFER ==========
        // Copy reconstructed data from iteration-specific array to final output array
        // Process each coil separately to maintain proper memory layout
        for (size_t iCHA = 0; iCHA < this->ncoils_; iCHA++)
            cudaMemcpyAsync(out->data() + inter_acc * data_dimensions[0] + stride_data * iCHA,
                            slice_view_out.get_data_ptr() + *(this->shots_per_time_.begin() + it) * data_dimensions[0] * iCHA,
                            *(this->shots_per_time_.begin() + it) * data_dimensions[0] * sizeof(float_complext),
                            cudaMemcpyDefault);

        // Return to original device context
        cudaSetDevice(cur_device);

    } // End of parallel processing loop
}
/**
 * @brief Adjoint operation: Non-Cartesian k-space data -> Cartesian image
 *
 * This function performs the adjoint (transpose) of the forward operation,
 * converting non-Cartesian k-space data back to Cartesian image domain.
 * Includes motion correction using deformation fields.
 *
 * @param in Input non-Cartesian k-space data
 * @param out Output Cartesian image data
 * @param accumulate Whether to accumulate results or overwrite output
 */
template <class REAL, unsigned int D>
void cuNonCartesianMOCOOperator_fc<REAL, D>::mult_MH(cuNDArray<complext<REAL>> *in, cuNDArray<complext<REAL>> *out,
                                                     bool accumulate)
{
    // ========== INPUT VALIDATION ==========
    // Validate input and output arrays are not null
    if (!in || !out)
    {
        throw std::runtime_error("cuNonCartesianSenseOperator::mult_MH : 0x0 input/output not accepted");
    }

    // Validate array dimensions match expected codomain and domain (reversed from forward)
    if (!in->dimensions_equal(&this->codomain_dims_) || !out->dimensions_equal(&this->domain_dims_))
    {
        throw std::runtime_error(
            "cuNonCartesianSenseOperator::mult_MH: input/output arrays do not match specified domain/codomains");
    }

    // ========== DIMENSION SETUP ==========
    // Adjoint operation: Non-Cartesian k-space -> Cartesian image
    std::vector<size_t> out_dimensions = *this->get_domain_dimensions();   // Cartesian image dimensions (output)
    std::vector<size_t> in_dimensions = *this->get_codomain_dimensions();  // Non-Cartesian data dimensions (input)

    // Extract key dimensions from input data
    auto RO = in->get_size(0);    // Readout points
    auto E1E2 = in->get_size(1);  // Encoding steps (phase encoding)
    auto CHA = in->get_size(2);   // Number of coils

    in_dimensions.pop_back(); // Remove coil dimension from input dimensions

    // ========== OUTPUT DIMENSION PREPARATION ==========
    int dims_orig = (out_dimensions.size());

    // Reduce to spatial dimensions only (remove extra dimensions beyond D)
    for (int ii = 0; ii < dims_orig - D; ii++)
    {
        out_dimensions.pop_back();
    }

    // Add respiratory/motion dimension for temporary storage
    out_dimensions.push_back(this->shots_per_time_.get_size(0)); // X Y Z R (respiratory states)

    // Create temporary array to combine coil data across motion states
    cuNDArray<complext<REAL>> tmp_coilCmb(&out_dimensions);

    // Remove respiratory dimension for final output
    out_dimensions.pop_back(); // Back to X Y Z

    auto out_slice_dims = out_dimensions;
    auto is4D = dims_orig == 4 ? true : false; // Check if we have time dimension

    // Calculate memory strides for data access
    auto stride_ch = std::accumulate(in_dimensions.begin(), in_dimensions.end(), 1,
                                     std::multiplies<size_t>()); // Stride for input channel data
    auto stride_out = std::accumulate(out_dimensions.begin(), out_dimensions.end(), 1,
                                      std::multiplies<size_t>()); // Stride for output spatial data

    // Clear output array if not accumulating results
    if (!accumulate)
    {
        clear(out);
    }

    // ========== DEVICE SETUP ==========
    int cur_device = in->get_device();
    cudaSetDevice(cur_device);

    // ========== FINAL DIMENSION CONFIGURATION ==========
    out_dimensions.push_back(this->ncoils_); // Add coil dimension for processing
    auto out_dimensions2 = out_dimensions;    // Copy for spatial-only operations
    out_dimensions2.pop_back();               // Remove coil dimension from copy
    in_dimensions.pop_back();                 // Remove interleave dimension from input

    GadgetronTimer timer("Reconstruct Sense");

    // ========== MOTION CORRECTION INITIALIZATION ==========
    //cuNonCartesianMOCOOperator<float, 3> mocoObj(this->convolutionType);
    gpuRegistration gr;

    // ========== TIME DIMENSION HANDLING ==========
    // Determine number of temporal frames to process
    size_t time_dims;
    if (is4D)
        time_dims = this->shots_per_time_.get_size(1); // Multiple time frames
    else
        time_dims = 1; // Single time frame

    auto moco_dims = this->shots_per_time_.get_size(0); // Number of motion states
    
    // ========== RECONSTRUCTION PROCESSING ==========
    auto out_store = out_dimensions; // Store original dimensions for restoration
    cuNDArray<complext<REAL>> moving_images;

    // Process each time frame separately
    for (size_t ito = 0; ito < time_dims; ito++)
    {
        // Create view into output array for this time frame
        auto slice_view_output = cuNDArray<complext<REAL>>(out_slice_dims, out->data() + stride_out * ito);

        // ========== PARALLEL MOTION STATE PROCESSING ==========
        // Process each motion state in parallel across available GPUs
        #pragma omp parallel for num_threads(std::min({eligibleGPUs.size(), this->shots_per_time_.get_size(0)})) shared(in, out, trajectory_, combinationWeights_, scaled_time_, fbins_) private(out_dimensions) ordered
        for (size_t it = 0; it < this->shots_per_time_.get_size(0); it++)
        {
            // Restore dimensions for this thread (modified by parallel processing)
            out_dimensions = out_store;
            cudaSetDevice(cur_device);

            // Distribute work across available GPUs
            auto gpuDevice = eligibleGPUs[it % eligibleGPUs.size()];

            // Create view into temporary coil-combined array for this motion state
            auto tmp_view = cuNDArray<complext<REAL>>(out_dimensions2, tmp_coilCmb.data() + stride_out * it);

            // ========== INPUT DATA EXTRACTION ==========
            // Set up dimensions for input data slice corresponding to this motion state
            auto in_dim_t = in_dimensions;
            in_dim_t.push_back(*(this->shots_per_time_.begin() + it + ito * moco_dims)); // Add shot count for this state
            in_dim_t.push_back(CHA); // Add coil dimension

            // Calculate accumulated offset to find correct input data slice
            auto inter_acc = std::accumulate(this->shots_per_time_.begin(), this->shots_per_time_.begin() + it + ito * moco_dims, 0);

            cuNDArray<complext<REAL>> tmp;

            // Extract the specific k-space data slice for this motion state and time point
            auto slice_view = crop<float_complext, 3>(uint64d3(0, inter_acc, 0),
                                                      uint64d3(RO, *(this->shots_per_time_.begin() + it + ito * moco_dims), this->ncoils_),
                                                      *in);

            // ========== RECONSTRUCTION SETUP ==========
            cudaSetDevice(gpuDevice);
            nhlbi_toolbox::reconstruction::noncartesian_reconstruction_fc nrfc(this->recon_params_);

            
            size_t indexDef = 0;
            if (forward_deformation_.size() == moco_dims)
                indexDef = it;
            else
                indexDef = it + ito * moco_dims;
            
            
                if (out_dimensions.size() == 4)
                    out_dimensions.pop_back();


                tmp.create(out_dimensions); // x y z

                if (cur_device != gpuDevice)
                {
                    
                    nrfc.reconstruct_todevice(slice_view, tmp_view, trajectory_[it + ito * moco_dims], (this->dcw_[it + ito * moco_dims]), *this->csm_.get(),
                                              *combinationWeights_, scaled_time_[it + ito * moco_dims], fbins_, gpuDevice, true);
                }
                else
                {
                    
                    nrfc.reconstruct(slice_view, tmp_view, trajectory_[it + ito * moco_dims], (this->dcw_[it + ito * moco_dims]), *this->csm_.get(),
                                     *combinationWeights_, scaled_time_[it + ito * moco_dims], fbins_);
                }
                cudaSetDevice(cur_device);

                
         
        }

        
        if (doMC_iter_ && this->counter % iteration_count == 0)
        {
            cudaSetDevice(cur_device);
            GDEBUG_STREAM("iteration_count :" << iteration_count);
            PythonFunction<hoNDArray<float>> register_images("registration_gadget_call", "registration_images");

            this->forward_deformation_.clear();
            this->backward_deformation_.clear();
            auto images_all = hoNDArray<std::complex<float>>(std::move(*boost::reinterpret_pointer_cast<hoNDArray<std::complex<float>>>(tmp_coilCmb.to_host())));
            float refP =this->get_refPhase();
            auto def = register_images(abs(images_all),refP,recon_params_.selectedDevices);
            auto deformation = cuNDArray<REAL>(def);
            auto inv_deformation = deformation;
            inv_deformation *= -1.0f;

            cuNDArray<REAL> output;

            output.create({recon_params_.rmatrixSize.x, recon_params_.rmatrixSize.y, recon_params_.rmatrixSize.z});
            crop<REAL, 5>(uint64d5((images_all.get_size(0) - recon_params_.rmatrixSize.x) / 2, (images_all.get_size(1) - recon_params_.rmatrixSize.y) / 2, (images_all.get_size(2) - recon_params_.rmatrixSize.z) / 2,0,0),
                       uint64d5(recon_params_.rmatrixSize.x, recon_params_.rmatrixSize.y, output.get_size(2),deformation.get_size(3),deformation.get_size(4)),
                       deformation,
                       output);
            deformation = output;

            crop<REAL, 5>(uint64d5((images_all.get_size(0) - recon_params_.rmatrixSize.x) / 2, (images_all.get_size(1) - recon_params_.rmatrixSize.y) / 2, (images_all.get_size(2) - recon_params_.rmatrixSize.z) / 2,0,0),
                       uint64d5(recon_params_.rmatrixSize.x, recon_params_.rmatrixSize.y, output.get_size(2),deformation.get_size(3),deformation.get_size(4)),
                       inv_deformation,
                       output);
                inv_deformation = output;

        

            auto defDims = *inv_deformation.get_dimensions();
            if (defDims[defDims.size() - 1] > 1)
                defDims.pop_back();

            auto stride = std::accumulate(defDims.begin(), defDims.end(), 1,
                                     std::multiplies<size_t>());

            std::vector<size_t> recon_dims = {images_all.get_size(0), images_all.get_size(1), 3, images_all.get_size(2)};

            for (auto ii = 0; ii < def.get_size(4); ii++)
            {
                auto defView = cuNDArray<float>(defDims, deformation.data() + stride * ii);
                auto intdefView = cuNDArray<float>(defDims, inv_deformation.data() + stride * ii);

                this->forward_deformation_.push_back(nhlbi_toolbox::utils::padDeformations(defView, recon_dims));
                this->backward_deformation_.push_back(nhlbi_toolbox::utils::padDeformations(intdefView, recon_dims));
            }
            
          
        }
        for (size_t it = 0; it < this->shots_per_time_.get_size(0); it++)
        {

            cudaSetDevice(cur_device);

            auto def2 = nhlbi_toolbox::utils::set_device(&(this->forward_deformation_)[it], cur_device);
            auto tmp_view = cuNDArray<complext<REAL>>(out_dimensions2, tmp_coilCmb.data() + stride_out * it); //  stride_out * ito * moco_dims +
            slice_view_output += gr.deform_image(&tmp_view, def2);
            }

        slice_view_output /= complext<REAL>((REAL)this->shots_per_time_.get_size(0), (REAL)0);
    }
    this->counter++;

    // nhlbi_toolbox::utils::write_gpu_nd_array<float_complext>(output, "/opt/data/gt_data/output_norm.complex");
}

template <class REAL, unsigned int D>
void cuNonCartesianMOCOOperator_fc<REAL, D>::def_to_back_vec(cuNDArray<REAL> *deformation)
{
    // ========== DIMENSION SETUP ==========
    std::vector<cuNDArray<float>> def;
    auto defDims = *deformation->get_dimensions();

    // Remove the last dimension if it's greater than 1 (typically the component dimension)
    if (defDims[defDims.size() - 1] > 1)
        defDims.pop_back();

    // Calculate stride for accessing individual deformation fields
    auto stride = std::accumulate(defDims.begin(), defDims.end(), 1,
                                  std::multiplies<size_t>());

    // ========== DEFORMATION FIELD EXTRACTION ==========
    // Extract each motion state's deformation field and store in backward_deformation_ vector
    for (auto ii = 0; ii < deformation->get_size(4); ii++)
    {
        GDEBUG_STREAM("ii:" << ii);
        // Create view into the 5D array for this motion state
        auto defView = cuNDArray<float>(defDims, deformation->data() + stride * ii);
        auto defview2 = defView;
        (this->backward_deformation_).push_back(defview2);
    }
    GDEBUG_STREAM("def size:" << (this->backward_deformation_).size());
}

/**
 * @brief Convert 5D deformation field array to vector of forward deformation fields
 *
 * This function takes a 5D deformation field array and splits it into individual
 * 3D deformation fields for each motion state. These forward deformation fields
 * are used to correct for motion by warping images from the reference frame
 * to the moving frame.
 *
 * @param deformation Input 5D deformation field array [X, Y, Z, components, motion_states]
 */
template <class REAL, unsigned int D>
void cuNonCartesianMOCOOperator_fc<REAL, D>::def_to_vec(cuNDArray<REAL> *deformation)
{
    // ========== DIMENSION SETUP ==========
    std::vector<cuNDArray<float>> def;
    auto defDims = *deformation->get_dimensions();

    // Remove the last dimension if it's greater than 1 (typically the component dimension)
    if (defDims[defDims.size() - 1] > 1)
        defDims.pop_back();

    // Calculate stride for accessing individual deformation fields
    auto stride = std::accumulate(defDims.begin(), defDims.end(), 1,
                                  std::multiplies<size_t>());

    // ========== DEFORMATION FIELD EXTRACTION ==========
    // Extract each motion state's deformation field and store in forward_deformation_ vector
    for (auto ii = 0; ii < deformation->get_size(4); ii++)
    {
        GDEBUG_STREAM("ii:" << ii);
        // Create view into the 5D array for this motion state
        auto defView = cuNDArray<float>(defDims, deformation->data() + stride * ii);
        auto defview2 = defView;
        (this->forward_deformation_).push_back(defview2);
    }
    GDEBUG_STREAM("def size:" << (this->forward_deformation_).size());
}
template <class REAL, unsigned int D>
void cuNonCartesianMOCOOperator_fc<REAL, D>::set_forward_deformation(std::vector<cuNDArray<REAL>> *forward_deformation)
{
    forward_deformation_ = *forward_deformation;
}
template <class REAL, unsigned int D>
void cuNonCartesianMOCOOperator_fc<REAL, D>::set_backward_deformation(std::vector<cuNDArray<REAL>> *backward_deformation)
{
    backward_deformation_ = *backward_deformation;
}
/**
 * @brief Preprocessing function to initialize reconstruction operators on all eligible GPUs
 *
 * This function prepares the reconstruction pipeline by:
 * - Validating input trajectories and dimensions
 * - Determining eligible GPU devices based on memory requirements
 * - Creating and preprocessing reconstruction objects for each GPU
 * - Setting up trajectory data on each device
 *
 * @param trajectory Vector of k-space trajectories for each time point
 */
template <class REAL, unsigned int D>
void cuNonCartesianMOCOOperator_fc<REAL, D>::preprocess(std::vector<cuNDArray<vector_td<REAL, D>>> &trajectory)
{
    using namespace nhlbi_toolbox::utils;

    // ========== INITIALIZATION ==========
    omp_set_max_active_levels(256); // Allow deep OpenMP nesting

    // Validate trajectory input
    if (&(*trajectory.begin()) == 0x0)
    {
        throw std::runtime_error("cuNonCartesianTSenseOperator_fc: cannot preprocess 0x0 trajectory.");
    }

    // ========== DIMENSION VALIDATION ==========
    boost::shared_ptr<std::vector<size_t>> domain_dims = this->get_domain_dimensions();
    if (domain_dims.get() == 0x0 || domain_dims->size() == 0)
    {
        throw std::runtime_error("cuNonCartesianTSenseOperator_fc::preprocess : operator domain dimensions not set");
    }

    // ========== GPU DEVICE SELECTION ==========
    // If no GPUs specified, automatically find suitable devices based on memory requirements
    if(eligibleGPUs.empty()){
        GDEBUG_STREAM("Eligible GPUS not set, should never be the case");

        // Calculate memory requirements for data arrays
        auto data_dims = *this->get_codomain_dimensions();
        auto dataSizeT = std::accumulate(data_dims.begin(), data_dims.end(), size_t(1), std::multiplies<size_t>()) * 4 * 2;
        auto dataSize = dataSizeT + dataSizeT / (2 * this->ncoils_) + 3 * dataSizeT / (2 * this->ncoils_); // add dcw and traj

        auto image_dims = *this->get_domain_dimensions();
        auto imageSize = std::accumulate(image_dims.begin(), image_dims.end(), size_t(1), std::multiplies<size_t>()) * 4 * 2;
        dataSize += imageSize;

        

        eligibleGPUs = nhlbi_toolbox::utils::FindCudaDevices(dataSize * 1);
    }
    int cur_device = trajectory[0].get_device();
    /*
    for (auto ii = 0; ii < eligibleGPUs.size(); ii++)
    {
        cudaSetDevice(eligibleGPUs[ii]);
        std::vector<nhlbi_toolbox::reconstruction::noncartesian_reconstruction_fc> temp_vector;
        for (auto it = 0; it < this->shots_per_time_.get_number_of_elements(); it++)
        {

            auto ttraj = set_device(&trajectory[it], eligibleGPUs[ii]);
            nhlbi_toolbox::reconstruction::noncartesian_reconstruction_fc nrfc(this->recon_params_);

            nrfc.preprocess(&ttraj);
            temp_vector.push_back(nrfc);
        }
        nrfc_vector.push_back(temp_vector);
    }
    */
    cudaSetDevice(cur_device);
    GDEBUG_STREAM("cur_device mocofc: " << cur_device);

    // eligibleGPUs.erase(std::remove(eligibleGPUs.begin(), eligibleGPUs.end(), cur_device), eligibleGPUs.end());

    trajectory_ = trajectory;
    is_preprocessed_ = true;
}

template <class REAL, unsigned int D>
void cuNonCartesianMOCOOperator_fc<REAL, D>::set_fbins(arma::fvec fbins)
{
    fbins_ = fbins;
}
template <class REAL, unsigned int D>
void cuNonCartesianMOCOOperator_fc<REAL, D>::set_combination_weights(cuNDArray<float_complext> *combinationWeights)
{
    combinationWeights_ = combinationWeights;
}
template <class REAL, unsigned int D>
void cuNonCartesianMOCOOperator_fc<REAL, D>::set_recon_params(reconParams rp)
{
    recon_params_ = rp;
    // nc_recon_fc_ = new nhlbi_toolbox::reconstruction::noncartesian_reconstruction_fc(recon_params);
}
template <class REAL, unsigned int D>
void cuNonCartesianMOCOOperator_fc<REAL, D>::set_scaled_time(std::vector<cuNDArray<REAL>> &scaled_time)
{
    scaled_time_ = scaled_time;
}
template <class REAL, unsigned int D>
void cuNonCartesianMOCOOperator_fc<REAL, D>::set_dofield_correction(bool flag)
{
    doConcomitantFieldCorraction_ = flag;
    // if(is_preprocessed_)
    // reprocess to spread overGPUs
}


template class EXPORTGPUPMRI cuNonCartesianMOCOOperator_fc<float, 3>;

