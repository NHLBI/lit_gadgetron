#include "noncartesian_reconstruction.h"
#include "real_utilities.h"
#include "core_defines.h"
#include "cuNDArray_blas.h"

using namespace Gadgetron;
namespace nhlbi_toolbox {
namespace reconstruction {
template <size_t D> noncartesian_reconstruction<D>::noncartesian_reconstruction(reconParams recon_params) {
    const char* conda_prefix = std::getenv("CONDA_PREFIX");
    if (conda_prefix) {
        python_path = std::filesystem::path(conda_prefix) / "share" / "gadgetron" / "python";
    } else {
        python_path = std::filesystem::path("/opt/conda/envs/gadgetron") / "share" / "gadgetron" / "python";
    }
    Gadgetron::initialize_python();
    Gadgetron::add_python_path(python_path.generic_string());
    // GDEBUG_STREAM("python_path.generic_string(): " << python_path.generic_string());

    unsigned int warp_size = cudaDeviceManager::Instance()->warp_size();
    this->recon_params = recon_params;
    resx = recon_params.fov.x / float(recon_params.ematrixSize.x);
    resy = recon_params.fov.y / float(recon_params.ematrixSize.y);
    resz = recon_params.fov.z / float(recon_params.ematrixSize.z);

    // GDEBUG_STREAM("rmatsize:" << recon_params.rmatrixSize.x);
    // GDEBUG_STREAM("rmatsize:" << recon_params.rmatrixSize.y);
    // GDEBUG_STREAM("rmatsize:" << recon_params.rmatrixSize.z);

    // GDEBUG_STREAM("ematsize:" << recon_params.ematrixSize.x);
    // GDEBUG_STREAM("ematsize:" << recon_params.ematrixSize.y);
    // GDEBUG_STREAM("ematsize:" << recon_params.ematrixSize.z);
    image_dims_.push_back(recon_params.ematrixSize.x);
    image_dims_.push_back(recon_params.ematrixSize.y);

    if (D == 3 && recon_params.ematrixSize.z != 1) // 3D imaging
    {

        if (recon_params.ematrixSize.z % warp_size != 0)
            image_dims_.push_back(warp_size * (recon_params.ematrixSize.z / warp_size + 1));
        else
            image_dims_.push_back(recon_params.ematrixSize.z);

        image_dims_os_.push_back(
            ((static_cast<size_t>(std::ceil(image_dims_[0] * recon_params.oversampling_factor_)) + warp_size - 1) /
             warp_size) *
            warp_size);
        image_dims_os_.push_back(
            ((static_cast<size_t>(std::ceil(image_dims_[1] * recon_params.oversampling_factor_)) + warp_size - 1) /
             warp_size) *
            warp_size);
        image_dims_os_.push_back(
            ((static_cast<size_t>(std::ceil(image_dims_[2] * recon_params.oversampling_factor_)) + warp_size - 1) /
             warp_size) *
            warp_size); // No oversampling is needed in the z-direction for SOS

        recon_dims = {image_dims_[0], image_dims_[1], image_dims_[2], recon_params.numberChannels};
        recon_dims_encodSpace = {image_dims_[0], image_dims_[1], recon_params.ematrixSize.z,
                                 recon_params.numberChannels}; // Cropped to size of Encoded Matrix
        recon_dims_reconSpace = {recon_params.rmatrixSize.x, recon_params.rmatrixSize.y, recon_params.rmatrixSize.z,
                                 recon_params.numberChannels}; // Cropped to size of Recon Matrix
    }
    if (D == 2 && recon_params.ematrixSize.z == 1) // 2D imaging may look for MRD-Header definitions
    {
        // image_dims_os_ = uint64d<D>(((static_cast<size_t>(std::ceil(image_dims_[0] *
        // recon_params.oversampling_factor_)) + warp_size - 1) / warp_size) * warp_size,
        //                           ((static_cast<size_t>(std::ceil(image_dims_[1] *
        //                           recon_params.oversampling_factor_)) + warp_size - 1) / warp_size) * warp_size);
        image_dims_os_.push_back(
            ((static_cast<size_t>(std::ceil(image_dims_[0] * recon_params.oversampling_factor_)) + warp_size - 1) /
             warp_size) *
            warp_size);
        image_dims_os_.push_back(
            ((static_cast<size_t>(std::ceil(image_dims_[1] * recon_params.oversampling_factor_)) + warp_size - 1) /
             warp_size) *
            warp_size);

        recon_dims = {image_dims_[0], image_dims_[1], recon_params.numberChannels};
        recon_dims_encodSpace = {image_dims_[0], image_dims_[1],
                                 recon_params.numberChannels}; // Cropped to size of Encoded Matrix
        recon_dims_reconSpace = {recon_params.rmatrixSize.x, recon_params.rmatrixSize.y,
                                 recon_params.numberChannels}; // Cropped to size of Recon Matrix

        // this->nfft_plan_ = NFFT<cuNDArray, float, D>::make_plan(from_std_vector<size_t, D>(image_dims_),
        // from_std_vector<size_t, D>(image_dims_os_), recon_params.kernel_width_, ConvolutionType::ATOMIC);
    }
    this->nfft_plan_ = NFFT<cuNDArray, float, D>::make_plan(from_std_vector<size_t, D>(image_dims_),
                                                            from_std_vector<size_t, D>(image_dims_os_),
                                                            recon_params.kernel_width_, ConvolutionType::ATOMIC);
    dcfO.oversampling_factor_ = recon_params.oversampling_factor_dcf_;
    dcfO.kernel_width_ = recon_params.kernel_width_dcf_;
    dcfO.iterations = recon_params.iterations_dcf;
    dcfO.useIterativeDCWEstimated = recon_params.useIterativeDCWEstimated;
}

template <size_t D>
template <typename T>
std::vector<cuNDArray<T>> noncartesian_reconstruction<D>::arraytovector(cuNDArray<T>* inputArray,
                                                                        std::vector<size_t> number_elements) {
    std::vector<cuNDArray<T>> vectorOut;

    for (auto iph = 0; iph < number_elements.size(); iph++) {
        auto str_phase = std::accumulate(number_elements.begin(), number_elements.begin() + iph, size_t(0));
        auto array_view = cuNDArray<T>({number_elements[iph]}, (*inputArray).get_data_ptr() + str_phase);
        vectorOut.push_back(std::move(array_view));
    }
    return std::move(vectorOut);
}

template <size_t D>
template <typename T>
std::vector<cuNDArray<T>> noncartesian_reconstruction<D>::arraytovector(cuNDArray<T>* inputArray,
                                                                        hoNDArray<size_t> number_elements) {
    std::vector<cuNDArray<T>> vectorOut;

    for (auto iph = 0; iph < number_elements.get_number_of_elements(); iph++) {

        auto str_phase = std::accumulate(number_elements.begin(), number_elements.begin() + iph, size_t(0));
        auto array_view = cuNDArray<T>({number_elements[iph]}, (*inputArray).get_data_ptr() + str_phase);
        vectorOut.push_back(array_view);
    }
    return std::move(vectorOut);
}

template <size_t D>
cuNDArray<float_complext>
noncartesian_reconstruction<D>::apply_gcc_matrix(cuNDArray<float_complext> images,
                                                 std::vector<hoNDArray<std::complex<float>>> mtx) {
    // GDEBUG_STREAM("python_path.generic_string(): " << python_path.generic_string());

    auto ims_host = hoNDArray<std::complex<float>>(
        std::move(*boost::reinterpret_pointer_cast<hoNDArray<std::complex<float>>>(images.to_host())));
    hoNDArray<std::complex<float>> ims_host_out;
    try {
        PythonFunction<hoNDArray<std::complex<float>>> apply_gcc_compress("gcc", "apply_gcc_compress");
        GDEBUG_STREAM("apply_gcc_compress loaded");
        for (size_t it = 0; it < mtx.size(); it++) {
            ims_host_out = apply_gcc_compress(&ims_host, &mtx[it], it);
            GDEBUG_STREAM("GCC Apply " << it << " done");
            ims_host = ims_host_out;
        }
    } catch (...) {
        GERROR_STREAM("Something broke");
    }
    return cuNDArray<float_complext>(hoNDArray<float_complext>(ims_host)); // Return the final compressed data
}
template <size_t D>
cuNDArray<float_complext> noncartesian_reconstruction<D>::estimate_gcc_matrix(cuNDArray<float_complext> images) {
    hoNDArray<std::complex<float>> compressed_data, compressed_data_a, mtxr, mtxra;
    hoNDArray<std::complex<float>> compressed_data_p, compressed_data_pa, mtxp, mtxpa;
    hoNDArray<std::complex<float>> compressed_data_z, compressed_data_za, mtxz, mtxza;

    auto ims_host = hoNDArray<std::complex<float>>(
        std::move(*boost::reinterpret_pointer_cast<hoNDArray<std::complex<float>>>(images.to_host())));
    std::vector<cuNDArray<float_complext>> mtx_vec;

    try {

        PythonFunction<hoNDArray<std::complex<float>>, hoNDArray<std::complex<float>>> gcc_compress("gcc",
                                                                                                    "gcc_compress");
        GDEBUG_STREAM("GCC loaded");
        auto ncc_def = ims_host.get_size(ims_host.get_number_of_dimensions() - 1);
        std::tie(compressed_data_a, mtxra) =
            gcc_compress(&ims_host, 24, 0, floor((ncc_def + recon_params.gcc_coils) / 2)); // First compression
        mtx_vec.push_back(cuNDArray<float_complext>(hoNDArray<float_complext>(mtxra)));
        // GDEBUG_STREAM("GCC Compress 1 done");
        // auto xx = apply_gcc_matrix(images,mtx_vec);
        ncc_def = compressed_data_a.get_size(compressed_data_a.get_number_of_dimensions() - 1);
        // floor((ncc_def+recon_params.gcc_coils)/2);
        std::tie(compressed_data_pa, mtxpa) =
            gcc_compress(&compressed_data_a, 24, 1, floor((ncc_def + recon_params.gcc_coils) / 2)); // First compression
        mtx_vec.push_back(cuNDArray<float_complext>(hoNDArray<float_complext>(mtxpa)));
        std::tie(compressed_data_za, mtxza) =
            gcc_compress(&compressed_data_pa, 24, 2, recon_params.gcc_coils); // First compression
        mtx_vec.push_back(cuNDArray<float_complext>(hoNDArray<float_complext>(mtxza)));

        GDEBUG_STREAM("GCC Compress 2 done");
    } catch (...) {
        GERROR_STREAM("Something broke");
    }

    GDEBUG_STREAM("mtxra dimensions: " << mtxra.get_size(0) << " x " << mtxra.get_size(1) << " y "
                                       << mtxra.get_size(2));
    set_mtx_vec(mtx_vec);
    return cuNDArray<float_complext>(hoNDArray<float_complext>(compressed_data_za)); // Return the final compressed data
}

template <size_t D>
template <typename T>
cuNDArray<T> noncartesian_reconstruction<D>::vectortoarray(std::vector<cuNDArray<T>>* inputArray) {
    // Only works with traj and dcf !!
    hoNDArray<size_t> num_ele_per_time = recon_params.shots_per_time;
    num_ele_per_time *= recon_params.RO;
    auto all_element = std::accumulate(num_ele_per_time.begin(), num_ele_per_time.end(), size_t(0));
    cuNDArray<T> arrayout(all_element);
    GDEBUG_STREAM("COPY" << num_ele_per_time.get_number_of_elements() << " " << all_element);
    for (auto iph = 0; iph < num_ele_per_time.get_number_of_elements(); iph++) {
        auto str_phase = std::accumulate(num_ele_per_time.begin(), num_ele_per_time.begin() + iph, size_t(0));
        GDEBUG_STREAM("COPY data" << iph << " " << str_phase << " " << all_element);
        cudaMemcpy(arrayout.get_data_ptr() + str_phase, (*inputArray)[iph].get_data_ptr(),
                   (*inputArray)[iph].get_number_of_elements() * sizeof(T), cudaMemcpyDefault);
    }
    return std::move(arrayout);
}

template <size_t D>
template <typename T>
cuNDArray<T> noncartesian_reconstruction<D>::crop_to_recondims(cuNDArray<T>& input) {

    cuNDArray<T> output;
    size_t NDim = input.get_number_of_dimensions();

    GDEBUG_STREAM("Input dimensions: ");
    for (size_t i = 0; i < input.get_number_of_dimensions(); ++i) {
        GDEBUG_STREAM("Dim " << i << ": " << input.get_size(i));
    }
    GDEBUG_STREAM("Number of dimensions: " << NDim);

    if (D == 2 && recon_params.ematrixSize.z == 1) {

        switch (NDim) {
        case 2:
            output.create({this->recon_dims_reconSpace[0], this->recon_dims_reconSpace[1]});
            crop<T, 2>(uint64d2((image_dims_[0] - this->recon_dims_reconSpace[0]) / 2,
                                (image_dims_[1] - this->recon_dims_reconSpace[1]) / 2),
                       uint64d2(this->recon_dims_reconSpace[0], this->recon_dims_reconSpace[1]), input, output);
            break;

        case 3:
            output.create({this->recon_dims_reconSpace[0], this->recon_dims_reconSpace[1], input.get_size(2)});
            crop<T, 3>(uint64d3((image_dims_[0] - this->recon_dims_reconSpace[0]) / 2,
                                (image_dims_[1] - this->recon_dims_reconSpace[1]) / 2,
                                0),
                       uint64d3(this->recon_dims_reconSpace[0], this->recon_dims_reconSpace[1], input.get_size(2)),
                       input, output);
            break;

        case 4:
            output.create(
                {this->recon_dims_reconSpace[0], this->recon_dims_reconSpace[1], input.get_size(2), input.get_size(3)});
            crop<T, 4>(uint64d4((image_dims_[0] - this->recon_dims_reconSpace[0]) / 2,
                                (image_dims_[1] - this->recon_dims_reconSpace[1]) / 2,
                                0, 0),
                       uint64d4(this->recon_dims_reconSpace[0], this->recon_dims_reconSpace[1],
                                input.get_size(2), input.get_size(3)),
                       input, output);
            break;

        default:
            GDEBUG_STREAM("crop_to_recondims is not working, unknow number of dimensions " << NDim);
        }
    }
    if (D == 3 && recon_params.ematrixSize.z != 1) {

        switch (NDim) {
        case 3:
            output.create(
                {this->recon_dims_reconSpace[0], this->recon_dims_reconSpace[1], this->recon_dims_reconSpace[2]});
            crop<T, 3>(uint64d3((image_dims_[0] - this->recon_dims_reconSpace[0]) / 2,
                                (image_dims_[1] - this->recon_dims_reconSpace[1]) / 2,
                                (image_dims_[2] - this->recon_dims_reconSpace[2]) / 2),
                       uint64d3(this->recon_dims_reconSpace[0], this->recon_dims_reconSpace[1],
                                this->recon_dims_reconSpace[2]),
                       input, output);
            break;

        case 4:
            output.create({this->recon_dims_reconSpace[0], this->recon_dims_reconSpace[1],
                           this->recon_dims_reconSpace[2], input.get_size(3)});
            crop<T, 4>(uint64d4((image_dims_[0] - this->recon_dims_reconSpace[0]) / 2,
                                (image_dims_[1] - this->recon_dims_reconSpace[1]) / 2,
                                (image_dims_[2] - this->recon_dims_reconSpace[2]) / 2, 0),
                       uint64d4(this->recon_dims_reconSpace[0], this->recon_dims_reconSpace[1],
                                this->recon_dims_reconSpace[2], input.get_size(3)),
                       input, output);
            break;

        case 5:
            output.create({this->recon_dims_reconSpace[0], this->recon_dims_reconSpace[1],
                           this->recon_dims_reconSpace[2], input.get_size(3), input.get_size(4)});
            crop<T, 5>(uint64d5((image_dims_[0] - this->recon_dims_reconSpace[0]) / 2,
                                (image_dims_[1] - this->recon_dims_reconSpace[1]) / 2,
                                (image_dims_[2] - this->recon_dims_reconSpace[2]) / 2, 0, 0),
                       uint64d5(this->recon_dims_reconSpace[0], this->recon_dims_reconSpace[1],
                                this->recon_dims_reconSpace[2], input.get_size(3), input.get_size(4)),
                       input, output);
            break;

        default:
            GDEBUG_STREAM("crop_to_recondims is not working, unknow number of dimensions " << NDim);
        }
    }
    return output;
}

template <size_t D>
boost::shared_ptr<cuNDArray<float_complext>>
noncartesian_reconstruction<D>::generateCSM(cuNDArray<float_complext>* channel_images) {
    auto CHA = channel_images->get_size(channel_images->get_number_of_dimensions() - 1); // Last dimension is chanels

    cuNDArray<float_complext> channel_images_cropped(recon_dims_encodSpace);
    crop<float_complext, 4>(uint64d4(0, 0, (recon_dims[2] - recon_dims_encodSpace[2]) / 2, 0),
                            uint64d4(recon_dims[0], recon_dims[1], recon_dims_encodSpace[2], CHA), channel_images,
                            channel_images_cropped);

    //  recon_dims = {image_dims_[0], image_dims_[1], CHA, E2}; // Cropped to size of Recon Matrix
    *channel_images = pad<float_complext, 4>(uint64d4(recon_dims[0], recon_dims[1], recon_dims[2], CHA),
                                             channel_images_cropped, float_complext(0));
    // auto temp = boost::make_shared<cuNDArray<float_complext>>(estimate_b1_map<float, 3>(channel_images_cropped));
    auto temp = boost::make_shared<cuNDArray<float_complext>>(
        nhlbi_toolbox::utils::estimateCoilmaps_slice(channel_images_cropped));

    auto csm = boost::make_shared<cuNDArray<float_complext>>(
        pad<float_complext, 4>(uint64d4(recon_dims[0], recon_dims[1], recon_dims[2], CHA), *temp, float_complext(0)));

    return csm;
}

template <size_t D>
boost::shared_ptr<cuNDArray<float_complext>>
noncartesian_reconstruction<D>::generateMcKenzieCSM(cuNDArray<float_complext>* channel_images) {
    // McKenzie et al. (Magn Reson Med2002;47:529-538.)
    cuNDArray<float> scale_image(recon_dims_encodSpace);
    auto tmp_csm =
        std::move(*boost::reinterpret_pointer_cast<hoNDArray<std::complex<float>>>((*channel_images).to_host()));
    auto temp = tmp_csm;
    auto rsos = *sum(abs_square(&temp).get(), D);
    sqrt_inplace(&rsos);
    auto cuda_rsos = cuNDArray<float>(rsos);

    for (size_t iCHA = 0; iCHA < recon_params.numberChannels; iCHA++) {
        cudaMemcpy(scale_image.get_data_ptr() + iCHA * cuda_rsos.get_number_of_elements(), cuda_rsos.get_data_ptr(),
                   (cuda_rsos).get_number_of_elements() * sizeof(float), cudaMemcpyDefault);
    }
    auto tmp_scale_image = std::move(*boost::reinterpret_pointer_cast<hoNDArray<float>>(scale_image.to_host()));
    tmp_csm /= tmp_scale_image;
    auto csm = boost::make_shared<cuNDArray<float_complext>>(hoNDArray<float_complext>(tmp_csm));

    return csm;
}

template <size_t D>
boost::shared_ptr<cuNDArray<float_complext>>
noncartesian_reconstruction<D>::generateEspiritCSM(cuNDArray<float_complext>* channel_images) {

    hoNDArray<std::complex<float>> ho_espirit_csm;
    auto channel_images_host = hoNDArray<std::complex<float>>(
        std::move(*boost::reinterpret_pointer_cast<hoNDArray<std::complex<float>>>(channel_images->to_host())));
    boost::shared_ptr<cuNDArray<float_complext>> espirit_csm;
    try {
        GDEBUG_STREAM("Test");
        PythonFunction<hoNDArray<std::complex<float>>> espirit_csm_calculation("coil_maps", "espirit_csm_calculation");
        GDEBUG_STREAM("Espirit Python");
        ho_espirit_csm = espirit_csm_calculation(&channel_images_host, 12, 3, 0.5,
                                                 0.02); // calib_width=12,kernel_width=3,crop=0.5,thresh=0.02
        GDEBUG_STREAM("GCC Compress 2 done");
    } catch (...) {
        GERROR_STREAM("Something broke");
    }
    espirit_csm = boost::make_shared<cuNDArray<float_complext>>(
        cuNDArray<float_complext>(hoNDArray<float_complext>(ho_espirit_csm)));
    return espirit_csm; // Return the final compressed data
}

template <size_t D>
boost::shared_ptr<cuNDArray<float_complext>>
noncartesian_reconstruction<D>::generateRoemerCSM(cuNDArray<float_complext>* channel_images) {
    auto CHA = channel_images->get_size(channel_images->get_number_of_dimensions() - 1); // Last dimension is chanels

    cuNDArray<float_complext> channel_images_cropped(recon_dims_reconSpace);
    boost::shared_ptr<cuNDArray<float_complext>> csm_new_pad;
    if (D == 3) {
        channel_images_cropped = crop_to_recondims(*channel_images);
        auto filtered_csm = std::move(
            *boost::reinterpret_pointer_cast<hoNDArray<std::complex<float>>>(channel_images_cropped.to_host()));
        for (auto ii = 0; ii < filtered_csm.get_size(3); ii++) {
            auto temp = hoNDArray<std::complex<float>>(filtered_csm(slice, slice, slice, ii));
            temp.squeeze();
            nhlbi_toolbox::utils::filterImagealongSlice(temp, get_kspace_filter_type("gaussian"), 100,
                                                        filtered_csm.get_size(2) / filtered_csm.get_size(1) * 30);

            nhlbi_toolbox::utils::filterImage(temp, get_kspace_filter_type("gaussian"), 100, 30);
            filtered_csm(slice, slice, slice, ii) = temp;
        }
        auto temp = filtered_csm;
        auto rsos = *sum(abs_square(&temp).get(), D);
        sqrt_inplace(&rsos);
        filtered_csm /= rsos;
        auto csm_new = cuNDArray<float_complext>(hoNDArray<float_complext>(filtered_csm));

        csm_new_pad = boost::make_shared<cuNDArray<float_complext>>(
            pad<float_complext, 4>(uint64d4(recon_dims[0], recon_dims[1], recon_dims[2], filtered_csm.get_size(3)),
                                   csm_new, float_complext(0)));
        //  recon_dims = {image_dims_[0], image_dims_[1], CHA, E2}; // Cropped to size of Recon Matrix
        *channel_images = pad<float_complext, 4>(uint64d4(recon_dims[0], recon_dims[1], recon_dims[2], CHA),
                                                 channel_images_cropped, float_complext(0));
    } else {
        channel_images_cropped = channel_images;
        auto filtered_csm = std::move(
            *boost::reinterpret_pointer_cast<hoNDArray<std::complex<float>>>(channel_images_cropped.to_host()));
        for (auto ii = 0; ii < filtered_csm.get_size(2); ii++) {
            auto temp = hoNDArray<std::complex<float>>(filtered_csm(slice, slice, ii));
            temp.squeeze();
            nhlbi_toolbox::utils::filterImage<2>(temp, get_kspace_filter_type("gaussian"), 100, 30);
            filtered_csm(slice, slice, ii) = temp;
        }
        // nhlbi_toolbox::utils::filterImage(filtered_csm, get_kspace_filter_type("gaussian"), 101, 30);
        auto temp = filtered_csm;
        auto rsos = *sum(abs_square(&temp).get(), D);
        sqrt_inplace(&rsos);
        filtered_csm /= rsos;
        csm_new_pad = boost::make_shared<cuNDArray<float_complext>>(hoNDArray<float_complext>(filtered_csm));
    }
    return csm_new_pad;
}

template <size_t D>
void noncartesian_reconstruction<D>::reconstruct(cuNDArray<float_complext>* data, cuNDArray<float_complext>* image,
                                                 cuNDArray<vector_td<float, D>>* traj, cuNDArray<float>* dcw) {

    //  GadgetronTimer timer("Reconstruct");
    auto RO = data->get_size(0);
    auto E1E2 = data->get_size(1);
    auto CHA = data->get_size(2);

    // if (!this->isprocessed)
    {
        // GadgetronTimer timer("Preprocess");
        this->nfft_plan_->preprocess(*traj, NFFT_prep_mode::C2NC);
        this->isprocessed = true;
    }
    auto data_dimensions = *data->get_dimensions();
    auto image_dimensions = *image->get_dimensions();
    if (CHA != 1) {
        data_dimensions.pop_back();  // remove CHA
        image_dimensions.pop_back(); // remove CHA
    }

    auto stride = std::accumulate(data_dimensions.begin(), data_dimensions.end(), 1,
                                  std::multiplies<size_t>()); // product of X,Y,and Z

    auto stride_results = std::accumulate(image_dimensions.begin(), image_dimensions.end(), 1,
                                          std::multiplies<size_t>()); // product of X,Y,and Z

    struct cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, data->get_device());
    // GDEBUG_STREAM("device space: " << float(cudaDeviceManager::Instance()->getFreeMemory(data->get_device())) /
    // float(std::pow(1024, 3))); GDEBUG_STREAM("data space: " << float(data->get_number_of_elements() * 4 * 2 * 8) /
    // float(std::pow(1024, 3))); // this is not working auto data_and_imageSize = float((stride * 4) * (2 * CHA + 1 +
    // 3) * std::pow(recon_params.oversampling_factor_, D) * 4 + (stride_results * 4) * (2 * CHA) *
    // std::pow(recon_params.oversampling_factor_, D) * 4) / float(std::pow(1024, 3)); GDEBUG_STREAM("data and image
    // space: " << float(data_and_imageSize)); // this is not working

    if(!recon_params.try_channel_gridding)
    {
        for (int iCHA = 0; iCHA < CHA; iCHA++) {
            auto data_view = cuNDArray<complext<float>>(data_dimensions, data->data() + stride * iCHA);
            auto results_view = cuNDArray<complext<float>>(image_dimensions, image->data() + stride_results * iCHA);

            this->nfft_plan_->compute(data_view, results_view, dcw, NFFT_comp_mode::BACKWARDS_NC2C);
        }
        cudaDeviceSynchronize();
    }else{
    
        try {
            this->nfft_plan_->compute(*data, *image, dcw, NFFT_comp_mode::BACKWARDS_NC2C);
            cudaDeviceSynchronize();
        } catch (const std::exception& e) {
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                GERROR_STREAM("CUDA error in deconstruct: " << cudaGetErrorString(err));
            }
            for (int iCHA = 0; iCHA < CHA; iCHA++) {
            auto data_view = cuNDArray<complext<float>>(data_dimensions, data->data() + stride * iCHA);
            auto results_view = cuNDArray<complext<float>>(image_dimensions, image->data() + stride_results * iCHA);

            this->nfft_plan_->compute(data_view, results_view, dcw, NFFT_comp_mode::BACKWARDS_NC2C);
            }
        cudaDeviceSynchronize();
            this->recon_params.try_channel_gridding=false;
            GDEBUG_STREAM("Try Channel gridding to false ");
            cudaDeviceSynchronize();
        } catch (...) {
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                GERROR_STREAM("Unknown CUDA error in deconstruct: " << cudaGetErrorString(err));
            }
            for (int iCHA = 0; iCHA < CHA; iCHA++) {
            auto data_view = cuNDArray<complext<float>>(data_dimensions, data->data() + stride * iCHA);
            auto results_view = cuNDArray<complext<float>>(image_dimensions, image->data() + stride_results * iCHA);
            this->nfft_plan_->compute(data_view, results_view, dcw, NFFT_comp_mode::BACKWARDS_NC2C);
            }
            cudaDeviceSynchronize();
            this->recon_params.try_channel_gridding=false;
        }
    }

}

template <size_t D>
void noncartesian_reconstruction<D>::reconstruct(cuNDArray<float_complext>* data, cuNDArray<float_complext>* image,
                                                 cuNDArray<vector_td<float, D>>* traj, cuNDArray<float>* dcw,
                                                 cuNDArray<float_complext>* csm) {

    //  GadgetronTimer timer("Reconstruct");
    auto RO = data->get_size(0);
    auto E1E2 = data->get_size(1);
    auto CHA = data->get_size(2);

    // if (!this->isprocessed)
    {
        // GadgetronTimer timer("Preprocess");
        this->nfft_plan_->preprocess(*traj, NFFT_prep_mode::NC2C);
        this->isprocessed = true;
    }

    auto out_dimensions = *image->get_dimensions();
    auto in_dimensions = *data->get_dimensions();

    if (CHA != 1 || csm->get_size(csm->get_number_of_dimensions() - 1) == CHA) {
        in_dimensions.pop_back(); // remove CHA
    }

    auto stride_ch = std::accumulate(in_dimensions.begin(), in_dimensions.end(), 1,
                                     std::multiplies<size_t>()); // product of X,Y,and Z

    auto stride_out = std::accumulate(out_dimensions.begin(), out_dimensions.end(), 1,
                                      std::multiplies<size_t>()); // product of X,Y,and Z

    auto out_view_ch=cuNDArray<float_complext>(out_dimensions, image->data());

    size_t numteams=std::min(size_t(2),size_t(CHA));
    //cuNDArray<float> test;
    //#pragma omp declare reduction(complex_add : cuNDArray<float> : omp_out += omp_in) \
    //    initializer(omp_priv = cuNDArray<float>(omp_orig))

    //#pragma omp target teams num_teams(numteams) 
    //#pragma omp distribute parallel for reduction(complex_add: test)
    for (size_t ich = 0; ich < CHA; ich++) {


        auto slice_view=cuNDArray<float_complext>(in_dimensions, data->get_data_ptr() + stride_ch * ich);
        auto tmpview = cuNDArray<float_complext>(out_dimensions);

        this->nfft_plan_->compute(&slice_view, tmpview, dcw, NFFT_comp_mode::BACKWARDS_NC2C);

        auto csm_view = cuNDArray<float_complext>(out_dimensions, csm->get_data_ptr() + stride_out * ich);
        tmpview *= *conj(&csm_view);
        out_view_ch += tmpview;
    }
    cudaDeviceSynchronize();
}

template <size_t D>
void noncartesian_reconstruction<D>::deconstruct(cuNDArray<float_complext>* images, cuNDArray<float_complext>* data,
                                                 cuNDArray<vector_td<float, D>>* traj, cuNDArray<float>* dcw) {

    // GadgetronTimer timer("Reconstruct");
    auto RO = data->get_size(0);
    auto E1E2 = data->get_size(1);
    auto CHA = data->get_size(2);

    // GDEBUG_STREAM("Channel: " << CHA);
    // GDEBUG_STREAM("E1E2:    " << E1E2);
    // GDEBUG_STREAM("RO:      " << RO);
    // GDEBUG_STREAM("X: " << images->get_size(0));
    // GDEBUG_STREAM("Y: " << images->get_size(1));
    // GDEBUG_STREAM("Z: " << images->get_size(2));
    // GDEBUG_STREAM("C: " << images->get_size(3));

    // if (!this->isprocessed)
    {
        this->nfft_plan_->preprocess(*traj, NFFT_prep_mode::C2NC);
        this->isprocessed = true;
    }
    this->nfft_plan_->compute(*images, *data, dcw, NFFT_comp_mode::FORWARDS_C2NC);
}

template <size_t D>
#include "sense_utilities.h"
void noncartesian_reconstruction<D>::deconstruct(cuNDArray<float_complext>* images, cuNDArray<float_complext>* data,
                                                 cuNDArray<vector_td<float, D>>* traj, cuNDArray<float>* dcw,
                                                 cuNDArray<float_complext>* csm) {

    // GadgetronTimer timer("Reconstruct");
    auto RO = data->get_size(0);
    auto E1E2 = data->get_size(1);
    auto CHA = data->get_size(2);

    // GDEBUG_STREAM("Channel: " << CHA);
    // GDEBUG_STREAM("E1E2:    " << E1E2);
    // GDEBUG_STREAM("RO:      " << RO);
    // if (!this->isprocessed)
    {
        this->nfft_plan_->preprocess(*traj, NFFT_prep_mode::C2NC);
        this->isprocessed = true;
    }

    auto data_dimensions = *data->get_dimensions();
    auto image_dimensions = *images->get_dimensions();

    if (csm->get_size(csm->get_number_of_dimensions() - 1) == CHA)
        data_dimensions.pop_back();

    auto stride = std::accumulate(data_dimensions.begin(), data_dimensions.end(), 1,
                                  std::multiplies<size_t>()); // product of X,Y,and Z

    auto stride_results = std::accumulate(image_dimensions.begin(), image_dimensions.end(), 1,
                                          std::multiplies<size_t>()); // product of X,Y,and Z

    struct cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, data->get_device());

    // cudaSetDevice(data->get_device());
    // GDEBUG_STREAM("Failed: now running in slower channel by channel mode");
    if(!recon_params.try_channel_gridding)
    {
        for (int iCHA = 0; iCHA < CHA; iCHA++) {
            auto data_view = cuNDArray<complext<float>>(data_dimensions, data->data() + stride * iCHA);
            auto csm_view = cuNDArray<complext<float>>(image_dimensions, csm->data() + stride_results * iCHA);
            cuNDArray<complext<float>> tmp_view(image_dimensions);
            tmp_view = *images;
            tmp_view *= csm_view;
            this->nfft_plan_->compute(&tmp_view, data_view, dcw, NFFT_comp_mode::FORWARDS_C2NC);
        }
        cudaDeviceSynchronize();
    }else{
    
        try {
            auto dims_csm = csm->get_dimensions();
            cuNDArray<float_complext> images_mult_csm(dims_csm);
            csm_mult_M<float, D>(images, &images_mult_csm, csm);
            this->nfft_plan_->compute(images_mult_csm, *data, dcw, NFFT_comp_mode::FORWARDS_C2NC);
            cudaDeviceSynchronize();
        } catch (const std::exception& e) {
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                GERROR_STREAM("CUDA error in deconstruct: " << cudaGetErrorString(err));
            }
            for (int iCHA = 0; iCHA < CHA; iCHA++) {
                auto data_view = cuNDArray<complext<float>>(data_dimensions, data->data() + stride * iCHA);
                auto csm_view = cuNDArray<complext<float>>(image_dimensions, csm->data() + stride_results * iCHA);
                cuNDArray<complext<float>> tmp_view(image_dimensions);
                tmp_view = *images;
                tmp_view *= csm_view;
                this->nfft_plan_->compute(&tmp_view, data_view, dcw, NFFT_comp_mode::FORWARDS_C2NC);
            }
            this->recon_params.try_channel_gridding=false;
            GDEBUG_STREAM("Try Channel gridding to false ");
            cudaDeviceSynchronize();
        } catch (...) {
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                GERROR_STREAM("Unknown CUDA error in deconstruct: " << cudaGetErrorString(err));
            }
            for (int iCHA = 0; iCHA < CHA; iCHA++) {
                auto data_view = cuNDArray<complext<float>>(data_dimensions, data->data() + stride * iCHA);
                auto csm_view = cuNDArray<complext<float>>(image_dimensions, csm->data() + stride_results * iCHA);
                cuNDArray<complext<float>> tmp_view(image_dimensions);
                tmp_view = *images;
                tmp_view *= csm_view;
                this->nfft_plan_->compute(&tmp_view, data_view, dcw, NFFT_comp_mode::FORWARDS_C2NC);
            }
            cudaDeviceSynchronize();
            this->recon_params.try_channel_gridding=false;
        }
    }

    // this->nfft_plan_->compute(*images, *data, dcw, NFFT_comp_mode::FORWARDS_C2NC);
}

template <size_t D>
std::tuple<cuNDArray<float_complext>, cuNDArray<vector_td<float, D>>, cuNDArray<float>, std::vector<size_t>>
noncartesian_reconstruction<D>::organize_data(std::vector<Core::Acquisition>* allAcq,
                                              std::vector<std::vector<size_t>> idx_phases)

{

    auto sumall = 0;
    std::vector<size_t> nelem_idx;
    for (auto iph = 0; iph < idx_phases.size(); iph++) {
        sumall += idx_phases[iph].size();
        nelem_idx.push_back(idx_phases[iph].size());
    }
    std::vector<size_t> data_dims = {recon_params.RO, sumall, recon_params.numberChannels};
    std::vector<size_t> traj_dims = {recon_params.RO, sumall};
    auto stride = std::accumulate(data_dims.begin(), data_dims.end() - 1, size_t(1), std::multiplies<size_t>());
    std::vector<size_t> number_elements;
    auto cutraj = cuNDArray<vector_td<float, D>>(traj_dims);
    auto cudcf = cuNDArray<float>(traj_dims);

    auto cuData = boost::make_shared<cuNDArray<float_complext>>(cuNDArray<float_complext>(data_dims));

    for (auto jj = 0; jj < idx_phases.size(); jj++) {
        auto str_phase = std::accumulate(nelem_idx.begin(), nelem_idx.begin() + jj, size_t(0));
        number_elements.push_back(0);
        for (auto idx_ph = 0; idx_ph < idx_phases[jj].size(); idx_ph++) {
            auto& [head, data, traj] = allAcq->at(idx_phases[jj][idx_ph]);
            for (size_t iCHA = 0; iCHA < recon_params.numberChannels; iCHA++) {
                cudaMemcpy(cuData.get()->get_data_ptr() + stride * iCHA + data.get_size(0) * (str_phase + idx_ph),
                           data.get_data_ptr() + data.get_size(0) * iCHA, // + totalnumInt,
                           data.get_size(0) * sizeof(complext<float>), cudaMemcpyDefault);
            }
            // traj does have a fourth bit but I am ignoring it its the place where initial DCF is stored and
            auto traj_dcw = nhlbi_toolbox::utils::separate_traj_and_dcw_2or3<D>(&(*traj));
            auto traj_sep = std::move(*std::get<0>(traj_dcw).get());
            auto dcw_sep = std::move(*std::get<1>(traj_dcw).get());

            number_elements[jj] = number_elements[jj] + data.get_size(0);
            cudaMemcpy(cutraj.get_data_ptr() + data.get_size(0) * (str_phase + idx_ph),
                       traj_sep.get_data_ptr(), // + totalnumInt,
                       recon_params.RO * sizeof(vector_td<float, D>), cudaMemcpyDefault);

            cudaMemcpy(cudcf.get_data_ptr() + data.get_size(0) * (str_phase + idx_ph),
                       dcw_sep.get_data_ptr(), // + totalnumInt,
                       data.get_size(0) * sizeof(float), cudaMemcpyDefault);
        }
    }

    cutraj.reshape(recon_params.RO * sumall);
    cudcf.reshape(recon_params.RO * sumall);

    return std::make_tuple(std::move(*cuData), std::move(cutraj), std::move(cudcf), number_elements);
}

template <size_t D>
std::tuple<cuNDArray<float_complext>, cuNDArray<vector_td<float, D>>, cuNDArray<float>>
noncartesian_reconstruction<D>::organize_data(std::vector<Core::Acquisition>* allAcq)

{
    std::vector<size_t> data_dims = {recon_params.RO, allAcq->size(), recon_params.numberChannels};
    std::vector<size_t> traj_dims = {recon_params.RO, allAcq->size()};
    auto stride = std::accumulate(data_dims.begin(), data_dims.end() - 1, size_t(1), std::multiplies<size_t>());

    auto cutraj = cuNDArray<vector_td<float, D>>(traj_dims);
    auto cudcw = cuNDArray<float>(traj_dims);

    auto cuData = boost::make_shared<cuNDArray<float_complext>>(cuNDArray<float_complext>(data_dims));

    for (auto jj = 0; jj < allAcq->size(); jj++) {
        auto& [head, data, traj] = allAcq->at(jj);
        for (size_t iCHA = 0; iCHA < recon_params.numberChannels; iCHA++) {

            cudaMemcpy(cuData.get()->get_data_ptr() + stride * iCHA + data_dims[0] * jj,
                       data.get_data_ptr() + data.get_size(0) * iCHA, // + totalnumInt,
                       data.get_size(0) * sizeof(complext<float>), cudaMemcpyDefault);
        }
        // traj does have a fourth bit but I am ignoring it its the place where initial DCF is stored and

        auto traj_dcw = nhlbi_toolbox::utils::separate_traj_and_dcw_2or3<D>(&(*traj));
        auto traj_sep = std::move(*std::get<0>(traj_dcw).get());
        auto dcw_sep = std::move(*std::get<1>(traj_dcw).get());

        cudaMemcpy(cutraj.get_data_ptr() + data_dims[0] * jj,
                   traj_sep.get_data_ptr(), // + totalnumInt,
                   recon_params.RO * sizeof(vector_td<float, D>), cudaMemcpyDefault);

        cudaMemcpy(cudcw.get_data_ptr() + data_dims[0] * jj,
                   dcw_sep.get_data_ptr(), // + totalnumInt,
                   recon_params.RO * sizeof(float), cudaMemcpyDefault);
    }

    // inefficiency need to copy traj back for DCF -> AJ promises in third person to fix it :D

    cutraj.reshape(recon_params.RO * allAcq->size());
    cudcw.reshape(recon_params.RO * allAcq->size());

    return std::make_tuple(std::move(*cuData), std::move(cutraj), std::move(cudcw));
}

template <size_t D>
std::tuple<cuNDArray<float_complext>, cuNDArray<vector_td<float, D>>, cuNDArray<float>>
noncartesian_reconstruction<D>::organize_data_hoNDArray(hoNDArray<float_complext>* hodata,
                                                        hoNDArray<vector_td<float, D>>* traj, hoNDArray<float>* dcw,
                                                        bool calculateDCF, bool calculateKPRECOND) {
    GDEBUG_STREAM("ORGANIZE DATA : data SHAPE RO " << hodata->get_size(0) << " INT " << hodata->get_size(1) << " CHA "
                                                   << hodata->get_size(2) << " TRAJ " << traj->get_size(0));

    size_t totalnumInt = hodata->get_size(1);
    cuNDArray<float> cudcw;
    std::vector<size_t> data_dims = {recon_params.RO, totalnumInt, recon_params.numberChannels};
    auto cuData = boost::make_shared<cuNDArray<float_complext>>(cuNDArray<float_complext>(data_dims));
    // auto cuData = boost::make_shared<cuNDArray<float_complext>>(cuNDArray<float_complext>(data));
    cudaMemcpy(cuData->get_data_ptr(), hodata->get_data_ptr(),
               cuData->get_number_of_elements() * sizeof(float_complext), cudaMemcpyHostToDevice);

    std::vector<size_t> flat_dims = {recon_params.RO * totalnumInt};
    auto cutraj = cuNDArray<vector_td<float, D>>(traj);
    cutraj.reshape(recon_params.RO * totalnumInt);
    cutraj.squeeze();

    if (calculateDCF) {
        GDEBUG_STREAM("Gadgetron is calculating DCF");
        if (calculateKPRECOND) {
            GDEBUG_STREAM("Estimation of DCF using kprecond");
            cudcw = estimate_kspace_precond(&cutraj);
        } else {
            GDEBUG_STREAM("Estimation of DCF using GT");
            cudcw = noncartesian_reconstruction::estimate_dcf(&cutraj);
        }
    } else {
        GDEBUG_STREAM("Gadgetron is not calculating DCF");
        cudcw = cuNDArray<float>(dcw);
    }
    return std::make_tuple(std::move(*cuData), std::move(cutraj), std::move(cudcw));
}

        template <size_t D>
        std::tuple<cuNDArray<vector_td<float, D>>,
                   cuNDArray<float>>
        noncartesian_reconstruction<D>::organize_traj_dcw_hoNDArray(
            hoNDArray<vector_td<float, D>> *traj,
            hoNDArray<float> *dcw,
            bool calculateDCF,
            bool calculateKPRECOND)
        {
            GDEBUG_STREAM("ORGANIZE DATA : traj SHAPE RO " << traj->get_size(0));
            
            cuNDArray<float> cudcw;

            auto cutraj = cuNDArray<vector_td<float, D>>(traj);
            //cutraj.reshape(recon_params.RO * totalnumInt);
            cutraj.squeeze();
            GDEBUG_STREAM("ORGANIZE DATA : traj SHAPE 0 " << cutraj.get_size(0));
            
            if (calculateDCF)
            {   
                GDEBUG_STREAM("Gadgetron is calculating DCF");
                if (calculateKPRECOND)
                {
                    GDEBUG_STREAM("Estimation of DCF using kprecond");
                    cudcw =estimate_kspace_precond(&cutraj);
                }
                else
                {
                    GDEBUG_STREAM("Estimation of DCF using GT");
                    cudcw = noncartesian_reconstruction::estimate_dcf(&cutraj);
                }
            }
            else
            {
                GDEBUG_STREAM("Gadgetron is not calculating DCF");
                cudcw = cuNDArray<float>(dcw);
            }
            return std::make_tuple(std::move(cutraj), std::move(cudcw));
        }
    
        template <size_t D>
        std::tuple<cuNDArray<float_complext>,
                   std::vector<cuNDArray<vector_td<float, D>>>,
                   std::vector<cuNDArray<float>>>
        noncartesian_reconstruction<D>::organize_data_vector(
            hoNDArray<float_complext> *data,
            hoNDArray<vector_td<float, D>> *traj,
            hoNDArray<float> *dcw,
            bool calculateDCF,
            bool calculateKPRECOND)
        {
            auto totalnumInt = std::accumulate(recon_params.shots_per_time.begin(), recon_params.shots_per_time.end(), size_t(0));
            GDEBUG_STREAM("DATA SIZE " << data->get_size(0) << " " << data->get_size(1) << data->get_size(2));
            GDEBUG_STREAM("EXPECTED DATA SIZE " << data->get_size(0) << " " << totalnumInt << " " << recon_params.numberChannels);

    auto [cuData, cutraj, cudcw] = organize_data_hoNDArray(data, traj, dcw, false);
    // auto cuData = boost::make_shared<cuNDArray<float_complext>>(Cudata);
    hoNDArray<size_t> num_ele_per_time = recon_params.shots_per_time;
    num_ele_per_time *= data->get_size(0); // Multiply per RO
    std::vector<cuNDArray<vector_td<float, D>>> cuTrajVec = arraytovector(&cutraj, num_ele_per_time);
    // std::vector<cuNDArray<float>> cuDCWVec = cuDCWVec=estimate_dcf(&cuTrajVec);
    std::vector<cuNDArray<float>> cuDCWVec;

    if (calculateDCF) {
        GDEBUG_STREAM("Gadgetron is calculating DCF");
        if (calculateKPRECOND) {
            GDEBUG_STREAM("Estimation of DCF using kprecond");
            cuDCWVec = estimate_kspace_precond_vector(&cuTrajVec);
        } else {
            GDEBUG_STREAM("Estimation of DCF using GT");
            cuDCWVec = estimate_dcf(&cuTrajVec);
        }

    } else {
        GDEBUG_STREAM("Gadgetron is not calculating DCF");
        cuDCWVec = arraytovector(&cudcw, num_ele_per_time);
    }

    return std::make_tuple(std::move(cuData), std::move(cuTrajVec), std::move(cuDCWVec));
}

template <size_t D>
cuNDArray<float> noncartesian_reconstruction<D>::estimate_dcf(cuNDArray<vector_td<float, 3>>* traj) {

    float kw_dcf = recon_params.kernel_width_dcf_;         // 1e-2
    float osf_dcf = recon_params.oversampling_factor_dcf_; // 1.5
    GDEBUG_STREAM("DCF parameters: kw " << kw_dcf << " os " << osf_dcf);
    auto dims_traj = *(traj->get_dimensions());
    std::vector<size_t> flat_dims = {traj->get_number_of_elements()};
    auto hoTraj = hoNDArray<vector_td<float, 3>>(
        std::move(*boost::reinterpret_pointer_cast<hoNDArray<vector_td<float, 3>>>((*traj).to_host())));

    std::vector<size_t> non_flat_dims = {recon_params.RO, traj->get_number_of_elements() / recon_params.RO};
    auto traj_view = hoNDArray<vector_td<float, 3>>(non_flat_dims, hoTraj.get_data_ptr());

    hoNDArray<float> hoflat_dcw = dcfO.estimate_DCF_slice(traj_view, image_dims_);

    float sum_all = asum(&hoflat_dcw);

    // std::vector<size_t> flat_dims = {traj->get_number_of_elements()};
    cuNDArray<float> flat_dcw = cuNDArray<float>(hoflat_dcw);

    // why is this consuming memory !
    float scale_factor = float(prod(from_std_vector<size_t, 3>(image_dims_os_))) / sum_all;
    flat_dcw *= scale_factor;
    flat_dcw.reshape(flat_dims);
    sqrt_inplace(&flat_dcw);
    return flat_dcw;
}

template <size_t D>
cuNDArray<float> noncartesian_reconstruction<D>::estimate_dcf(cuNDArray<vector_td<float, 3>>* traj,
                                                              cuNDArray<float>* dcf_in) {
    float kw_dcf = recon_params.kernel_width_dcf_;         // 1e-2
    float osf_dcf = recon_params.oversampling_factor_dcf_; // 1.5
    GDEBUG_STREAM("DCF parameters: kw " << kw_dcf << " os " << osf_dcf);
    auto dims_traj = *(traj->get_dimensions());

    auto hoTraj = hoNDArray<vector_td<float, 3>>(
        std::move(*boost::reinterpret_pointer_cast<hoNDArray<vector_td<float, 3>>>((*traj).to_host())));
    auto hodcw = hoNDArray<float>(std::move(*boost::reinterpret_pointer_cast<hoNDArray<float>>((*dcf_in).to_host())));
    std::vector<size_t> non_flat_dims = {recon_params.RO, traj->get_number_of_elements() / recon_params.RO};
    auto traj_view = hoNDArray<vector_td<float, 3>>(non_flat_dims, hoTraj.get_data_ptr());
    auto dcw_view = hoNDArray<float>(non_flat_dims, hodcw.get_data_ptr());

    hoNDArray<float> hoflat_dcw = dcfO.estimate_DCF_slice(traj_view, dcw_view, image_dims_);
    float sum_all = asum(&hoflat_dcw);

    std::vector<size_t> flat_dims = {traj->get_number_of_elements()};
    cudaSetDevice(traj->get_device());
    cuNDArray<float> flat_dcw = cuNDArray<float>(hoflat_dcw);

    // why is this consuming memory !
    float scale_factor = float(prod(from_std_vector<size_t, 3>(image_dims_os_))) / sum_all;
    flat_dcw *= scale_factor;
    flat_dcw.reshape(flat_dims);
    sqrt_inplace(&flat_dcw);
    return flat_dcw;
}

template <size_t D>
cuNDArray<float> noncartesian_reconstruction<D>::estimate_dcf(cuNDArray<vector_td<float, 2>>* traj,
                                                              cuNDArray<float>* dcf_in) {
    auto dims_traj = *(traj->get_dimensions());

    //            auto hoTraj = hoNDArray<vector_td<float,
    //            2>>(std::move(*boost::reinterpret_pointer_cast<hoNDArray<vector_td<float, 2>>>((*traj).to_host())));

    std::vector<size_t> non_flat_dims = {recon_params.RO, traj->get_number_of_elements() / recon_params.RO};
    auto traj_view = cuNDArray<vector_td<float, 2>>(non_flat_dims, traj->get_data_ptr());

    auto hoflat_dcw = dcfO.estimate_DCF(*traj, *dcf_in, image_dims_);
    float sum_all = asum(&hoflat_dcw);

    std::vector<size_t> flat_dims = {traj->get_number_of_elements()};
    cudaSetDevice(traj->get_device());

    auto flat_dcw = cuNDArray<float>(hoflat_dcw);
    float scale_factor = float(prod(from_std_vector<size_t, 2>(image_dims_os_))) / sum_all;
    // float scale_factor = float(prod(from_std_vector<size_t, 2>(image_dims_os_)));
    flat_dcw *= scale_factor;
    flat_dcw.reshape(flat_dims);
    sqrt_inplace(&flat_dcw);
    return flat_dcw;
}
template <size_t D>
cuNDArray<float> noncartesian_reconstruction<D>::estimate_dcf(cuNDArray<vector_td<float, 2>>* traj) {
    // auto dims_traj = *(traj->get_dimensions());

    //            auto hoTraj = hoNDArray<vector_td<float,
    //            2>>(std::move(*boost::reinterpret_pointer_cast<hoNDArray<vector_td<float, 2>>>((*traj).to_host())));

    // std::vector<size_t> non_flat_dims = {recon_params.RO, traj->get_number_of_elements() / recon_params.RO};
    // auto traj_view = cuNDArray<vector_td<float, 2>>(non_flat_dims, traj->get_data_ptr());

    auto hoflat_dcw = dcfO.estimate_DCF(*traj, image_dims_);
    float sum_all = asum(&hoflat_dcw);
    cudaSetDevice(traj->get_device());

    std::vector<size_t> flat_dims = {traj->get_number_of_elements()};
    auto flat_dcw = cuNDArray<float>(hoflat_dcw);

    float scale_factor = float(prod(from_std_vector<size_t, 2>(image_dims_os_))) / sum_all;
    flat_dcw *= scale_factor;
    flat_dcw.reshape(flat_dims);
    sqrt_inplace(&flat_dcw);
    return flat_dcw;
}

template <size_t D>
cuNDArray<float> noncartesian_reconstruction<D>::estimate_dcf(cuNDArray<vector_td<float, 3>>* traj,
                                                              std::vector<size_t> number_elements) {
    cudaSetDevice(traj->get_device());

    auto cudcw = cuNDArray<float>((traj)->get_number_of_elements());

    auto hoTraj = hoNDArray<vector_td<float, 3>>(
        std::move(*boost::reinterpret_pointer_cast<hoNDArray<vector_td<float, 3>>>(traj->to_host())));

    for (auto ii = 0; ii < number_elements.size(); ii++) {
        std::vector<size_t> non_flat_dims = {
            recon_params.RO,
            number_elements[ii] / recon_params.RO}; // this is needed because we haven't reworked DCF estimation
        auto str_phase = std::accumulate(number_elements.begin(), number_elements.begin() + ii, size_t(0));

        auto traj_view = hoNDArray<vector_td<float, 3>>(non_flat_dims, hoTraj.get_data_ptr() + str_phase);

        auto hoflat_dcw = dcfO.estimate_DCF_slice(traj_view, image_dims_);
        float sum_all = asum(&hoflat_dcw);

        cudaMemcpy(cudcw.get_data_ptr() + str_phase,
                   hoflat_dcw.get_data_ptr(), // + totalnumInt,
                   number_elements[ii] * sizeof(float), cudaMemcpyDefault);

        auto dcf_view = cuNDArray<float>({number_elements[ii]}, cudcw.get_data_ptr() + str_phase);
        float scale_factor = float(prod(from_std_vector<size_t, 3>(image_dims_os_))) / sum_all;
        dcf_view *= scale_factor;
        sqrt_inplace(&dcf_view);
    }

            return cudcw;
        }

        template <size_t D> 
        cuNDArray<float> noncartesian_reconstruction<D>::estimate_dcf_special(cuNDArray<vector_td<float, 3>> *traj)
        {
            cudaSetDevice(traj->get_device());
            GDEBUG_STREAM("------Special ESTIMATE DCF -------");
            auto cudcw = cuNDArray<float>((traj)->get_number_of_elements());

            auto hoTraj = hoNDArray<vector_td<float, 3>>(std::move(*boost::reinterpret_pointer_cast<hoNDArray<vector_td<float, 3>>>(traj->to_host())));
            float sum_all_all = 0;
            hoNDArray<size_t> number_elements = recon_params.shots_per_time;
            for (auto ii = 0; ii < number_elements.get_number_of_elements(); ii++)
            {
                std::vector<size_t> non_flat_dims = {recon_params.RO, number_elements[ii]}; // this is needed because we haven't reworked DCF estimation
                auto str_phase = size_t(recon_params.RO) * std::accumulate(number_elements.begin(), number_elements.begin() + ii, size_t(0));

                auto traj_view = hoNDArray<vector_td<float, 3>>(non_flat_dims, hoTraj.get_data_ptr() + str_phase);

                auto hoflat_dcw = dcfO.estimate_DCF_slice(traj_view, image_dims_);
                float sum_all = asum(&hoflat_dcw);
                sum_all_all+=sum_all;
                cudaMemcpy(cudcw.get_data_ptr() + str_phase,
                           hoflat_dcw.get_data_ptr(), // + totalnumInt,
                           number_elements[ii] *size_t(recon_params.RO)* sizeof(float), cudaMemcpyDefault);

                auto dcf_view = cuNDArray<float>({number_elements[ii]*size_t(recon_params.RO)}, cudcw.get_data_ptr() + str_phase);
                
                //dcf_view *= scale_factor;
                GDEBUG_STREAM("Sum DCF for segment " << ii << " is " << sum_all);    
            }
            float scale_factor = float(prod(from_std_vector<size_t, 3>(image_dims_os_)))/sum_all_all;  // / sum_all;
            GDEBUG_STREAM("Sum DCF total is " << sum_all_all << "scale_factor" << scale_factor);   
            cudcw*=scale_factor;
            sqrt_inplace(&cudcw);
            return cudcw;
        }

template <size_t D>
std::vector<cuNDArray<float>>
noncartesian_reconstruction<D>::estimate_dcf(std::vector<cuNDArray<vector_td<float, 2>>>* traj) {
    std::vector<cuNDArray<float>> cudcw;

    for (auto ii = 0; ii < (traj)->size(); ii++) {
        cudcw.push_back(std::move(estimate_dcf(&(traj->at(ii)))));
    }

    return cudcw;
}

template <size_t D>
void noncartesian_reconstruction<D>::apply_gcc_compress(cuNDArray<float_complext>& images,
                                                        cuNDArray<float_complext> mtx, size_t dim) {

    // cuNDArray<complext<float>> DATA(*images.get_dimensions());
    // DATA = images;
    Gadgetron::timeswitch3D(&images);
    cuNDFFT<float>::instance()->ifft3(&images);
    Gadgetron::timeswitch3D(&images);

    // auto ffts = cuFFTPlan<float_complext>(3,std::vector<size_t>({0,1,2}));
    // ffts.ifft3c(DATA);

    CC(images, mtx, dim);
    // ffts.fft3c(coil_compl);
    Gadgetron::timeswitch3D(&images);
    cuNDFFT<float>::instance()->fft3(&images);
    Gadgetron::timeswitch3D(&images);
    // return coil_compl;
}

template <size_t D>
void noncartesian_reconstruction<D>::CC(cuNDArray<float_complext>& DATA, cuNDArray<float_complext> mtx, size_t dim) {
    auto ncc = mtx.get_size(1);

    if (dim == 1) {
        DATA = permute<float_complext>(&DATA, {1, 0, 2, 3});
    }
    if (dim == 2) {
        DATA = permute<float_complext>(&DATA, {2, 0, 1, 3});
    }

    auto Nx = DATA.get_size(0);
    auto Ny = DATA.get_size(1);
    auto Nz = DATA.get_size(2);
    auto Nc = DATA.get_size(3);

    Gadgetron::timeswitch1D(&DATA);
    cuNDFFT<float>::instance()->ifft1(&DATA);
    Gadgetron::timeswitch1D(&DATA);
    // cuNDFFT<float_complext>::instance()->timeswitch(&DATA,0);
    // cuNDFFT<float_complext>::instance()->fft1(&DATA);
    // cuNDFFT<float_complext>::instance()->timeswitch(&DATA,0);
    auto rdims = std::vector<size_t>({Nx, Ny, Nz, ncc});

    cuNDArray<float_complext> res(rdims);
    fill(&res, float_complext(0.0f, 0.0f));

    DATA = permute<float_complext>(&DATA, {1, 2, 3, 0});
    res = permute<float_complext>(&res, {1, 2, 3, 0});

    for (size_t n = 0; n < Nx; n++) {

        auto tmpc = cuNDArray<float_complext>({Ny * Nz, Nc}, DATA.get_data_ptr() + n * Ny * Nz * Nc);
        auto mtx_view = cuNDArray<float_complext>({Nc, ncc}, mtx.get_data_ptr() + n * Nc * ncc);
        auto res_view = cuNDArray<float_complext>({Ny * Nz, ncc}, res.get_data_ptr() + n * Ny * Nz * ncc);

        cublasHandle_t handle;
        cublasCreate(&handle);
        const float_complext alpha = float_complext(1.0f, 1.0f);
        const float_complext beta = float_complext(0.0f, 0.0f);

        cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, Ny * Nz, ncc, Nc, reinterpret_cast<const cuComplex*>(&alpha),
                    reinterpret_cast<const cuComplex*>(tmpc.get_data_ptr()), Ny * Nz,
                    reinterpret_cast<const cuComplex*>(mtx_view.get_data_ptr()), Nc,
                    reinterpret_cast<const cuComplex*>(&beta), reinterpret_cast<cuComplex*>(res_view.get_data_ptr()),
                    Ny * Nz);

        cublasDestroy(handle);
    }
    DATA = permute<float_complext>(&res, {3, 0, 1, 2});

    Gadgetron::timeswitch1D(&DATA);
    cuNDFFT<float>::instance()->fft1(&DATA);
    Gadgetron::timeswitch1D(&DATA);

    if (dim == 1) {
        DATA = permute<float_complext>(&DATA, {1, 0, 2, 3});
    }
    if (dim == 2) {
        DATA = permute<float_complext>(&DATA, {1, 2, 0, 3});
    }

    // return res;
}

template <size_t D>
std::vector<cuNDArray<float>>
noncartesian_reconstruction<D>::estimate_dcf(std::vector<cuNDArray<vector_td<float, 3>>>* traj) {
    std::vector<cuNDArray<float>> cudcw;
    hoNDArray<size_t> num_ele_per_time = recon_params.shots_per_time;
    auto all_element = std::accumulate(num_ele_per_time.begin(), num_ele_per_time.end(), size_t(0));
    GDEBUG_STREAM("NumberOfelement" << num_ele_per_time.get_number_of_elements() << " " << all_element);
    for (auto ii = 0; ii < (traj)->size(); ii++) {
        cudcw.push_back(std::move(estimate_dcf(&(traj->at(ii)))));
        float scale_factor = sqrt(float(num_ele_per_time[ii]) / float(all_element));
        // GDEBUG_STREAM("SCALE_FACTOR"<<scale_factor);
        cudcw[ii] *= scale_factor;
        // auto hoflat_dcw =
        // hoNDArray<float>(std::move(*boost::reinterpret_pointer_cast<hoNDArray<float>>(cudcw.at(ii).to_host())));
        // float sum_all = asum(&hoflat_dcw);

        // float scale_factor = float(prod(from_std_vector<size_t, 3>(image_dims_os_))) / sum_all;
        // cudcw[ii] *= scale_factor;
        // sqrt_inplace(&cudcw[ii]);
    }

    return cudcw;
}

template <size_t D>
std::vector<cuNDArray<float>>
noncartesian_reconstruction<D>::estimate_dcf(std::vector<cuNDArray<vector_td<float, 3>>>* traj,
                                             std::vector<cuNDArray<float>>* dcf_in) {
    std::vector<cuNDArray<float>> cudcw;
    hoNDArray<size_t> num_ele_per_time = recon_params.shots_per_time;
    auto all_element = std::accumulate(num_ele_per_time.begin(), num_ele_per_time.end(), size_t(0));
    GDEBUG_STREAM("NumberOfelement" << num_ele_per_time.get_number_of_elements() << " " << all_element);
    for (auto ii = 0; ii < (traj)->size(); ii++) {
        cudcw.push_back(std::move(estimate_dcf(&(traj->at(ii)), &(dcf_in->at(ii)))));
        float scale_factor = sqrt(float(num_ele_per_time[ii]) / float(all_element));
        GDEBUG_STREAM("SCALE_FACTOR" << scale_factor);
        cudcw[ii] *= scale_factor;
        // auto hoflat_dcw =
        // hoNDArray<float>(std::move(*boost::reinterpret_pointer_cast<hoNDArray<float>>(cudcw.at(ii).to_host())));
        // float sum_all = asum(&hoflat_dcw);

        // float scale_factor = float(prod(from_std_vector<size_t, 3>(image_dims_os_))) / sum_all;
        // cudcw[ii] *= scale_factor;
        // sqrt_inplace(&cudcw[ii]);
    }

    return cudcw;
}

template <size_t D>
cuNDArray<float> noncartesian_reconstruction<D>::estimate_kspace_precond(cuNDArray<vector_td<float, D>>* traj) {
    GadgetronTimer timer("Estimate_kspace_precond:");
    hoNDArray<vector_td<float, D>> hoTraj = hoNDArray<vector_td<float, D>>(
        std::move(*boost::reinterpret_pointer_cast<hoNDArray<vector_td<float, D>>>((*traj).to_host())));

    float lamda = recon_params.kernel_width_dcf_;          // 1e-2
    float osf_dcf = recon_params.oversampling_factor_dcf_; // 1.5
    size_t me_x = recon_params.omatrixSize.x;
    size_t me_y = recon_params.omatrixSize.y;
    size_t me_z = recon_params.omatrixSize.z;

    hoNDArray<float> hoTraj_float = hoNDArray<float>({D, (hoTraj).get_size(0)});
    for (size_t i = 0; i < hoTraj.get_number_of_elements(); i++) {
        vector_td<float, D> pts = hoTraj[i];
        for (size_t d = 0; d < D; d++) {
            hoTraj_float[i * D + d] = pts[d];
        }
    }

    GDEBUG_STREAM("Required Original matrix size to be set (recon_params.omatrixSize) correctly X"
                  << me_x << " Y " << me_y << " Z " << me_z);
    GDEBUG_STREAM("Warning using kernel_width dcf as lambda " << lamda << " OSF_dcf " << osf_dcf);
    // GDEBUG_STREAM("HOtraj" << hoTraj.get_number_of_dimensions() << " " << hoTraj.get_size(0) << " " <<
    // hoTraj.get_size(1)  << " " <<hoTraj.get_number_of_elements()); GDEBUG_STREAM("HOtraj" <<
    // hoTraj_float.get_number_of_dimensions() << " " << hoTraj_float.get_size(0) << " " << hoTraj_float.get_size(1)  <<
    // " " <<hoTraj_float.get_number_of_elements());

    hoNDArray<float> hodcw;
    try {
        PythonFunction<hoNDArray<float>> kspace_precond("kspace_preconditioning", "estimate_k_precond");
        GDEBUG_STREAM("kspace_precond start");
        hodcw = kspace_precond(hoTraj_float, me_x, me_y, me_z, lamda, osf_dcf);
        GDEBUG_STREAM("kspace_precond end ");
    } catch (...) {
        GERROR_STREAM("Something broke");
        hodcw.create(hoTraj.get_number_of_elements());
        hodcw.fill(1);
    }
    return cuNDArray<float>(hodcw); // Return the final compressed data
}
template <size_t D>
std::vector<cuNDArray<float>>
noncartesian_reconstruction<D>::estimate_kspace_precond_vector(std::vector<cuNDArray<vector_td<float, D>>>* traj) {
    std::vector<cuNDArray<float>> cudcw;
    for (auto ii = 0; ii < (traj)->size(); ii++) {
        cudcw.push_back(std::move(estimate_kspace_precond(&(traj->at(ii)))));
    }

    return cudcw;
}

template class noncartesian_reconstruction<2>;
template class noncartesian_reconstruction<3>;
template cuNDArray<float_complext> noncartesian_reconstruction<2>::crop_to_recondims(cuNDArray<float_complext>& input);
template cuNDArray<float_complext> noncartesian_reconstruction<3>::crop_to_recondims(cuNDArray<float_complext>& input);
template cuNDArray<float> noncartesian_reconstruction<2>::crop_to_recondims(cuNDArray<float>& input);
template cuNDArray<float> noncartesian_reconstruction<3>::crop_to_recondims(cuNDArray<float>& input);
template std::vector<cuNDArray<float>>
noncartesian_reconstruction<2>::arraytovector(cuNDArray<float>* inputArray, std::vector<size_t> number_elements);
template std::vector<cuNDArray<floatd2>>
noncartesian_reconstruction<2>::arraytovector(cuNDArray<floatd2>* inputArray, std::vector<size_t> number_elements);
template std::vector<cuNDArray<float>>
noncartesian_reconstruction<3>::arraytovector(cuNDArray<float>* inputArray, std::vector<size_t> number_elements);
template std::vector<cuNDArray<floatd3>>
noncartesian_reconstruction<3>::arraytovector(cuNDArray<floatd3>* inputArray, std::vector<size_t> number_elements);
template std::vector<cuNDArray<float_complext>>
noncartesian_reconstruction<2>::arraytovector(cuNDArray<float_complext>* inputArray,
                                              std::vector<size_t> number_elements);

template std::vector<cuNDArray<float>> noncartesian_reconstruction<2>::arraytovector(cuNDArray<float>* inputArray,
                                                                                     hoNDArray<size_t> number_elements);
template std::vector<cuNDArray<floatd2>>
noncartesian_reconstruction<2>::arraytovector(cuNDArray<floatd2>* inputArray, hoNDArray<size_t> number_elements);
template std::vector<cuNDArray<float>> noncartesian_reconstruction<3>::arraytovector(cuNDArray<float>* inputArray,
                                                                                     hoNDArray<size_t> number_elements);
template std::vector<cuNDArray<floatd3>>
noncartesian_reconstruction<3>::arraytovector(cuNDArray<floatd3>* inputArray, hoNDArray<size_t> number_elements);

template cuNDArray<vector_td<float, 3>>
noncartesian_reconstruction<3>::vectortoarray(std::vector<cuNDArray<vector_td<float, 3>>>* inputArray);
template cuNDArray<float> noncartesian_reconstruction<3>::vectortoarray(std::vector<cuNDArray<float>>* inputArray);
template cuNDArray<float> noncartesian_reconstruction<2>::vectortoarray(std::vector<cuNDArray<float>>* inputArray);
} // namespace reconstruction
} // namespace nhlbi_toolbox