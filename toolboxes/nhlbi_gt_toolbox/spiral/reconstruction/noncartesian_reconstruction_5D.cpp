#include "noncartesian_reconstruction_5D.h"
#include "cuNlcgSolver.h"
using namespace Gadgetron;
namespace nhlbi_toolbox
{
    namespace reconstruction
    {
        cuNDArray<float_complext> noncartesian_reconstruction_5D::reconstruct(
            cuNDArray<float_complext> *data,
            std::vector<cuNDArray<floatd3>> *traj,
            std::vector<cuNDArray<float>> *dcw,
            boost::shared_ptr<cuNDArray<float_complext>> csm,
            bool crop_image)
        {
            // nhlbi_toolbox::utils::enable_peeraccess();
            auto data_dims = *data->get_dimensions();
            auto stride = std::accumulate(data_dims.begin(), data_dims.end() - 1, size_t(1), std::multiplies<size_t>());
            cudaSetDevice(data->get_device());

            // prep data and dcw
            for (auto ii = 0; ii < dcw->size(); ii++)
            {
                auto inter_acc = std::accumulate(recon_params.shots_per_time.begin(), recon_params.shots_per_time.begin() + (ii), size_t(0)) * data_dims[0];

                for (auto iCHA = 0; iCHA < recon_params.numberChannels; iCHA++)
                {
                    auto dataview = cuNDArray<complext<float>>((*dcw)[ii].get_dimensions(), data->data() + inter_acc + stride * iCHA);
                    dataview *= (*dcw)[ii];
                }
            }
            auto E_ = boost::shared_ptr<cuNonCartesianTSenseOperator<float, 3>>(new cuNonCartesianTSenseOperator<float, 3>(ConvolutionType::ATOMIC));
            auto D_ = boost::shared_ptr<cuCgPreconditioner<float_complext>>(new cuCgPreconditioner<float_complext>());
            
            cuGpBbSolver<float_complext> solver_;

            solver_.set_max_iterations(recon_params.iterations);

            recon_dims = {image_dims_[0], image_dims_[1], image_dims_[2], recon_params.shots_per_time.get_size(0), recon_params.shots_per_time.get_size(1)};

            cuNDArray<float_complext> reg_image(recon_dims);
            E_->set_recon_params(recon_params);
            E_->set_shots_per_time(recon_params.shots_per_time);
            E_->setup(from_std_vector<size_t, 3>(image_dims_), from_std_vector<size_t, 3>(image_dims_os_), recon_params.kernel_width_);
            E_->set_codomain_dimensions(&data_dims);
            E_->set_domain_dimensions(&recon_dims);
            E_->set_csm(csm);
            E_->set_dcw(*dcw);
            E_->preprocess(*traj);

            // auto x0 = boost::make_shared<cuNDArray<float_complext>>(cuNDArray<float_complext>(recon_dims));
            // E_->mult_MH(data, x0.get());
            solver_.set_encoding_operator(E_);

            solver_.set_tc_tolerance(recon_params.tolerance);
            solver_.set_output_mode(decltype(solver_)::OUTPUT_VERBOSE);
            // solver_.set_x0(x0);

            boost::shared_ptr<cuNDArray<float>> _precon_weights;
            boost::shared_ptr<cuNDArray<float_complext>> precon_weights;

            _precon_weights = sum(abs_square(csm.get()).get(), 3);
            recon_dims = {image_dims_[0], image_dims_[1], recon_params.rmatrixSize.z}; // Cropped to size of Recon Matrix
            cuNDArray<float> _precon_weights_cropped = this->crop_to_recondims<float>(*_precon_weights);
            reciprocal_sqrt_inplace(&_precon_weights_cropped);
            // reciprocal_sqrt_inplace(_precon_weights.get());
precon_weights = boost::make_shared<cuNDArray<float_complext>>(pad<float_complext, 3>(uint64d3(recon_dims[0], recon_dims[1], image_dims_[2]),
                                                                                                  *real_to_complex<float_complext>(&_precon_weights_cropped), float_complext(0)));

            GDEBUG_STREAM("Reseting prerecon weigths from inf to zero");
            auto ho_prerecon = hoNDArray<std::complex<float>>(std::move(*boost::reinterpret_pointer_cast<hoNDArray<std::complex<float>>>((*precon_weights).to_host())));
            auto ho_prereconr = real(ho_prerecon);
            auto ho_prereconi = imag(ho_prerecon);

            std::replace(ho_prereconr.begin(),ho_prereconr.end(),INFINITY,0.0f);
            std::replace(ho_prereconi.begin(),ho_prereconi.end(),INFINITY,0.0f);
            auto ho_prereconri = *real_imag_to_complex<float_complext>(&ho_prereconr,&ho_prereconi);
            precon_weights = boost::make_shared<cuNDArray<float_complext>>(hoNDArray<float_complext>(ho_prereconri));                                                                                                 

            D_->set_weights(precon_weights);
            solver_.set_preconditioner(D_);

            recon_dims = {image_dims_[0], image_dims_[1], image_dims_[2], recon_params.shots_per_time.get_size(0), recon_params.shots_per_time.get_size(1), recon_params.numberChannels};
            recon_dims.pop_back();

            boost::shared_ptr<cuPartialDerivativeOperator<float_complext, 5>>
                Rt2(new cuPartialDerivativeOperator<float_complext, 5>(4));

            Rt2->set_weight(recon_params.lambda_time2);

            Rt2->set_domain_dimensions(&recon_dims);
            Rt2->set_codomain_dimensions(&recon_dims);
            solver_.add_regularization_operator(Rt2, recon_params.norm);

            boost::shared_ptr<cuPartialDerivativeOperator<float_complext, 5>>
                Rt(new cuPartialDerivativeOperator<float_complext, 5>(3));

            Rt->set_weight(recon_params.lambda_time);

            Rt->set_domain_dimensions(&recon_dims);
            Rt->set_codomain_dimensions(&recon_dims);
            solver_.add_regularization_operator(Rt, recon_params.norm);

            boost::shared_ptr<cuPartialDerivativeOperator<float_complext, 5>>
                Rx(new cuPartialDerivativeOperator<float_complext, 5>(0));
            Rx->set_weight(recon_params.lambda_spatial_imoco);
            Rx->set_domain_dimensions(&recon_dims);
            Rx->set_codomain_dimensions(&recon_dims);

            boost::shared_ptr<cuPartialDerivativeOperator<float_complext, 5>>
                Ry(new cuPartialDerivativeOperator<float_complext, 5>(1));
            Ry->set_weight(recon_params.lambda_spatial_imoco);
            Ry->set_domain_dimensions(&recon_dims);
            Ry->set_codomain_dimensions(&recon_dims);

            boost::shared_ptr<cuPartialDerivativeOperator<float_complext, 5>>
                Rz(new cuPartialDerivativeOperator<float_complext, 5>(2));
            Rz->set_weight(recon_params.lambda_spatial_imoco * (resx * resx) / (resz * resz));
            Rz->set_domain_dimensions(&recon_dims);
            Rz->set_codomain_dimensions(&recon_dims);

            // TV->set_domain_dimensions(&recon_dims);
            solver_.add_regularization_operator(Rx, recon_params.norm);
            solver_.add_regularization_operator(Ry, recon_params.norm);
            solver_.add_regularization_operator(Rz, recon_params.norm);

            GDEBUG_STREAM("Data_device:" << data->get_device());
            GDEBUG_STREAM("gpus_input_possible[0]:" << recon_params.selectedDevices[0]<< " [1] if exist" << recon_params.selectedDevices[1]);
            solver_.set_gpus(recon_params.selectedDevices);
            cudaSetDevice(data->get_device());
            reg_image = *solver_.solve(data);

            cuNDArray<float_complext> images_cropped;
            if (crop_image)
            {
                images_cropped = this->crop_to_recondims<float_complext>(reg_image); // Recon Size (3D+t2+t1)

            }
            else
            {
                images_cropped=reg_image;
            }

            // de-prep data
            for (auto ii = 0; ii < dcw->size(); ii++)
            {
                auto inter_acc = std::accumulate(recon_params.shots_per_time.begin(), recon_params.shots_per_time.begin() + (ii), size_t(0)) * data_dims[0];

                for (auto iCHA = 0; iCHA < data->get_size(data->get_number_of_dimensions() - 1); iCHA++)
                {
                    auto dataview = cuNDArray<complext<float>>((*dcw)[ii].get_dimensions(), data->data() + inter_acc + stride * iCHA);
                    dataview /= (*dcw)[ii];
                }
            }
            return images_cropped;
        }

        cuNDArray<float_complext> noncartesian_reconstruction_5D::reconstruct_nlcg(
            cuNDArray<float_complext> *data,
            std::vector<cuNDArray<floatd3>> *traj,
            std::vector<cuNDArray<float>> *dcw,
            boost::shared_ptr<cuNDArray<float_complext>> csm)
        {
            // nhlbi_toolbox::utils::enable_peeraccess();
            auto data_dims = *data->get_dimensions();
            auto stride = std::accumulate(data_dims.begin(), data_dims.end() - 1, size_t(1), std::multiplies<size_t>());
            cudaSetDevice(data->get_device());

            // prep data and dcw
            for (auto ii = 0; ii < dcw->size(); ii++)
            {
                auto inter_acc = std::accumulate(recon_params.shots_per_time.begin(), recon_params.shots_per_time.begin() + (ii), size_t(0)) * data_dims[0];

                for (auto iCHA = 0; iCHA < recon_params.numberChannels; iCHA++)
                {
                    auto dataview = cuNDArray<complext<float>>((*dcw)[ii].get_dimensions(), data->data() + inter_acc + stride * iCHA);
                    dataview *= (*dcw)[ii];
                }
            }
            auto E_ = boost::shared_ptr<cuNonCartesianTSenseOperator<float, 3>>(new cuNonCartesianTSenseOperator<float, 3>(ConvolutionType::ATOMIC));
            auto D_ = boost::shared_ptr<cuCgPreconditioner<float_complext>>(new cuCgPreconditioner<float_complext>());

            cuNlcgSolver<float_complext> solver_;

            solver_.set_max_iterations(recon_params.iterations);

            recon_dims = {image_dims_[0], image_dims_[1], image_dims_[2], recon_params.shots_per_time.get_size(0), recon_params.shots_per_time.get_size(1)};

            cuNDArray<float_complext> reg_image(recon_dims);
            E_->set_recon_params(recon_params);
            E_->set_shots_per_time(recon_params.shots_per_time);
            E_->setup(from_std_vector<size_t, 3>(image_dims_), from_std_vector<size_t, 3>(image_dims_os_), recon_params.kernel_width_);
            E_->set_codomain_dimensions(&data_dims);
            E_->set_domain_dimensions(&recon_dims);
            E_->set_csm(csm);
            E_->set_dcw(*dcw);
            E_->preprocess(*traj);

            // auto x0 = boost::make_shared<cuNDArray<float_complext>>(cuNDArray<float_complext>(recon_dims));
            // E_->mult_MH(data, x0.get());
            solver_.set_encoding_operator(E_);

            solver_.set_tc_tolerance(recon_params.tolerance);
            solver_.set_output_mode(decltype(solver_)::OUTPUT_VERBOSE);
            // solver_.set_x0(x0);

            boost::shared_ptr<cuNDArray<float>> _precon_weights;
            boost::shared_ptr<cuNDArray<float_complext>> precon_weights;

            _precon_weights = sum(abs_square(csm.get()).get(), 3);
            recon_dims = {image_dims_[0], image_dims_[1], recon_params.rmatrixSize.z}; // Cropped to size of Recon Matrix
            cuNDArray<float> _precon_weights_cropped = this->crop_to_recondims<float>(*_precon_weights);
            reciprocal_sqrt_inplace(&_precon_weights_cropped);
            // reciprocal_sqrt_inplace(_precon_weights.get());
precon_weights = boost::make_shared<cuNDArray<float_complext>>(pad<float_complext, 3>(uint64d3(recon_dims[0], recon_dims[1], image_dims_[2]),
                                                                                                  *real_to_complex<float_complext>(&_precon_weights_cropped), float_complext(0)));

            GDEBUG_STREAM("Reseting prerecon weigths from inf to zero");
            auto ho_prerecon = hoNDArray<std::complex<float>>(std::move(*boost::reinterpret_pointer_cast<hoNDArray<std::complex<float>>>((*precon_weights).to_host())));
            auto ho_prereconr = real(ho_prerecon);
            auto ho_prereconi = imag(ho_prerecon);

            std::replace(ho_prereconr.begin(),ho_prereconr.end(),INFINITY,0.0f);
            std::replace(ho_prereconi.begin(),ho_prereconi.end(),INFINITY,0.0f);
            auto ho_prereconri = *real_imag_to_complex<float_complext>(&ho_prereconr,&ho_prereconi);
            precon_weights = boost::make_shared<cuNDArray<float_complext>>(hoNDArray<float_complext>(ho_prereconri));                                                                                                 

            D_->set_weights(precon_weights);
            solver_.set_preconditioner(D_);

            recon_dims = {image_dims_[0], image_dims_[1], image_dims_[2], recon_params.shots_per_time.size(), recon_params.numberChannels};
            recon_dims.pop_back();

            boost::shared_ptr<cuPartialDerivativeOperator<float_complext, 4>>
                Rt(new cuPartialDerivativeOperator<float_complext, 4>(3));

            Rt->set_weight(recon_params.lambda_time);

            Rt->set_domain_dimensions(&recon_dims);
            Rt->set_codomain_dimensions(&recon_dims);
            solver_.add_regularization_operator(Rt, recon_params.norm);

            boost::shared_ptr<cuPartialDerivativeOperator<float_complext, 5>>
                Rt2(new cuPartialDerivativeOperator<float_complext, 5>(4));

            Rt2->set_weight(recon_params.lambda_time2);

            Rt2->set_domain_dimensions(&recon_dims);
            Rt2->set_codomain_dimensions(&recon_dims);
            solver_.add_regularization_operator(Rt2, recon_params.norm);

            boost::shared_ptr<cuPartialDerivativeOperator<float_complext, 4>>
                Rx(new cuPartialDerivativeOperator<float_complext, 4>(0));
            Rx->set_weight(recon_params.lambda_spatial);
            Rx->set_domain_dimensions(&recon_dims);
            Rx->set_codomain_dimensions(&recon_dims);

            boost::shared_ptr<cuPartialDerivativeOperator<float_complext, 4>>
                Ry(new cuPartialDerivativeOperator<float_complext, 4>(1));
            Ry->set_weight(recon_params.lambda_spatial);
            Ry->set_domain_dimensions(&recon_dims);
            Ry->set_codomain_dimensions(&recon_dims);

            boost::shared_ptr<cuPartialDerivativeOperator<float_complext, 4>>
                Rz(new cuPartialDerivativeOperator<float_complext, 4>(2));
            Rz->set_weight(recon_params.lambda_spatial * (resx * resx) / (resz * resz));
            Rz->set_domain_dimensions(&recon_dims);
            Rz->set_codomain_dimensions(&recon_dims);

            // TV->set_domain_dimensions(&recon_dims);
            solver_.add_regularization_operator(Rx, recon_params.norm);
            solver_.add_regularization_operator(Ry, recon_params.norm);
            solver_.add_regularization_operator(Rz, recon_params.norm);
            cudaSetDevice(data->get_device());
            reg_image = *solver_.solve(data);
            cuNDArray<float_complext> images_cropped = this->crop_to_recondims<float_complext>(reg_image); // Recon Size (3D+t)

            // de-prep data
            for (auto ii = 0; ii < dcw->size(); ii++)
            {
                auto inter_acc = std::accumulate(recon_params.shots_per_time.begin(), recon_params.shots_per_time.begin() + (ii), size_t(0)) * data_dims[0];

                for (auto iCHA = 0; iCHA < data->get_size(data->get_number_of_dimensions() - 1); iCHA++)
                {
                    auto dataview = cuNDArray<complext<float>>((*dcw)[ii].get_dimensions(), data->data() + inter_acc + stride * iCHA);
                    dataview /= (*dcw)[ii];
                }
            }
            return images_cropped;
        }

        cuNDArray<float_complext> noncartesian_reconstruction_5D::reconstructiMOCO_avg_image(
            cuNDArray<float_complext> *data,
            std::vector<cuNDArray<floatd3>> *traj,
            std::vector<cuNDArray<float>> *dcw,
            cuNDArray<float_complext> avg_images,
            boost::shared_ptr<cuNDArray<float_complext>> csm,
            float referencePhase)
        {
            // // nhlbi_toolbox::utils::enable_peeraccess();

            int selectedDevice = nhlbi_toolbox::utils::selectCudaDevice();
            if (!recon_params.selectedDevices.empty()){
                selectedDevice=recon_params.selectedDevices.back();
            }
            cudaSetDevice(selectedDevice);
            auto deformations = register_images_gpu(avg_images, referencePhase);
            auto deformation = std::get<0>(deformations);
            auto inv_deformation = std::get<1>(deformations);
            avg_images.clear();

            auto data_dims = *data->get_dimensions();
            auto stride = std::accumulate(data_dims.begin(), data_dims.end() - 1, size_t(1), std::multiplies<size_t>());
            cudaSetDevice(data->get_device());

            // prep data and dcw - doing this data save in memory to prevent data from being affected by recon.
            hoNDArray<float_complext> hodata(*data->get_dimensions());
            cudaMemcpy(hodata.get_data_ptr(), data->get_data_ptr(), data->get_number_of_elements() * sizeof(float_complext), cudaMemcpyDeviceToHost);

            hoNDArray<float_complext> hocsm(*csm->get_dimensions());
            cudaMemcpy(hocsm.get_data_ptr(), csm->get_data_ptr(), csm->get_number_of_elements() * sizeof(float_complext), cudaMemcpyDeviceToHost);

            arma::fvec fbins(1);
            fbins.ones();
            std::vector<cuNDArray<float>> scaled_time_vec;
            // prep data and dcw
            // prep data and dcw - doing this data save in memory to prevent data from being affected by recon.
            
            for (auto ii = 0; ii < dcw->size(); ii++)
            {
                auto inter_acc = std::accumulate(recon_params.shots_per_time.begin(), recon_params.shots_per_time.begin() + (ii), size_t(0)) * data_dims[0];
                auto tmp = cuNDArray<float>(*(*dcw)[ii].get_dimensions());
                fill(&tmp, float(0.0));
                scaled_time_vec.push_back(tmp);
                for (auto iCHA = 0; iCHA < recon_params.numberChannels; iCHA++)
                {
                    auto dataview = cuNDArray<complext<float>>((*dcw)[ii].get_dimensions(), data->data() + inter_acc + stride * iCHA);
                    dataview *= (*dcw)[ii];
                }
            }

            std::vector<size_t> cwdims = {image_dims_[0], image_dims_[1], image_dims_[2], 1};
            cuNDArray<float_complext> padded_cw(cwdims);
            fill(&padded_cw, float_complext(1.0, 0.0));
            padded_cw.squeeze();

            recon_dims = {image_dims_[0], image_dims_[1], image_dims_[2], recon_params.shots_per_time.get_size(1)};

            cudaSetDevice(data->get_device());

            recon_dims = {image_dims_[0], image_dims_[1], image_dims_[2], recon_params.shots_per_time.get_size(1)};

            auto E_ = boost::shared_ptr<cuNonCartesianMOCOOperator_fc<float, 3>>(new cuNonCartesianMOCOOperator_fc<float, 3>(ConvolutionType::ATOMIC));
            auto D_ = boost::shared_ptr<cuCgPreconditioner<float_complext>>(new cuCgPreconditioner<float_complext>());

            nhlbi_toolbox::cuGpBbSolver<float_complext> solver_;
            // cuSbcCgSolver<float_complext> solver_;

            solver_.set_max_iterations(recon_params.iterations_imoco);
            // solver_.set_max_outer_iterations(recon_params.iterations);
            // solver_.set_max_inner_iterations(recon_params.iterations_inner);
            cuNDArray<float_complext> reg_image(recon_dims);

            E_->set_shots_per_time(recon_params.shots_per_time);
            E_->setup(from_std_vector<size_t, 3>(image_dims_), from_std_vector<size_t, 3>(image_dims_os_), recon_params.kernel_width_);
            E_->set_codomain_dimensions(&data_dims);
            E_->set_domain_dimensions(&recon_dims);
            E_->set_csm(csm);
            E_->set_dcw(*dcw);
            E_->set_recon_params(recon_params);
            E_->set_combination_weights(&padded_cw);
            E_->set_scaled_time(scaled_time_vec);
            E_->set_fbins(fbins);
            if (!recon_params.doMC_iter)
            {
                E_->set_forward_deformation(&deformation);
                E_->set_backward_deformation(&inv_deformation);
            }
            E_->set_doMC_iter(recon_params.doMC_iter);
            E_->set_iteration_count(recon_params.iteration_count_moco);
            E_->set_refPhase(referencePhase);
            E_->set_eligibleGPUs(recon_params.selectedDevices);
            // cudaSetDevice(padded_def[0].get_device());

            // E_->set_forward_deformation(&padded_def);
            // E_->set_backward_deformation(&padded_invdef);
            cudaSetDevice(data->get_device());

            // do everything before preprocess
            E_->preprocess(*traj);

            auto x0 = boost::make_shared<cuNDArray<float_complext>>(cuNDArray<float_complext>(recon_dims));
            E_->mult_MH(data, x0.get());
            solver_.set_encoding_operator(E_);

            solver_.set_tc_tolerance(recon_params.tolerance);
            solver_.set_output_mode(decltype(solver_)::OUTPUT_VERBOSE);
            solver_.set_x0(x0);

            boost::shared_ptr<cuNDArray<float>> _precon_weights;
            boost::shared_ptr<cuNDArray<float_complext>> precon_weights;

            _precon_weights = sum(abs_square(csm.get()).get(), 3);
            recon_dims = {image_dims_[0], image_dims_[1], recon_params.rmatrixSize.z}; // Cropped to size of Recon Matrix

            cuNDArray<float> _precon_weights_cropped = this->crop_to_recondims<float>(*_precon_weights);

            reciprocal_sqrt_inplace(&_precon_weights_cropped);
            // reciprocal_sqrt_inplace(_precon_weights.get());
            precon_weights = boost::make_shared<cuNDArray<float_complext>>(pad<float_complext, 3>(uint64d3(recon_dims[0], recon_dims[1], image_dims_[2]),
                                                                                                  *real_to_complex<float_complext>(&_precon_weights_cropped), float_complext(0)));

            GDEBUG_STREAM("Reseting prerecon weigths from inf to zero");
            auto ho_prerecon = hoNDArray<std::complex<float>>(std::move(*boost::reinterpret_pointer_cast<hoNDArray<std::complex<float>>>((*precon_weights).to_host())));
            auto ho_prereconr = real(ho_prerecon);
            auto ho_prereconi = imag(ho_prerecon);

            std::replace(ho_prereconr.begin(),ho_prereconr.end(),INFINITY,0.0f);
            std::replace(ho_prereconi.begin(),ho_prereconi.end(),INFINITY,0.0f);
            auto ho_prereconri = *real_imag_to_complex<float_complext>(&ho_prereconr,&ho_prereconi);
            precon_weights = boost::make_shared<cuNDArray<float_complext>>(hoNDArray<float_complext>(ho_prereconri));                                                                                                  
            _precon_weights->clear();
            _precon_weights_cropped.clear();

            D_->set_weights(precon_weights);
            solver_.set_preconditioner(D_);

            recon_dims = {image_dims_[0], image_dims_[1], image_dims_[2], recon_params.shots_per_time.get_size(1)};

            boost::shared_ptr<cuPartialDerivativeOperator<float_complext, 4>>
                Rx(new cuPartialDerivativeOperator<float_complext, 4>(0));
            Rx->set_weight(recon_params.lambda_spatial_imoco);
            Rx->set_domain_dimensions(&recon_dims);
            Rx->set_codomain_dimensions(&recon_dims);

            boost::shared_ptr<cuPartialDerivativeOperator<float_complext, 4>>
                Ry(new cuPartialDerivativeOperator<float_complext, 4>(1));
            Ry->set_weight(recon_params.lambda_spatial_imoco);
            Ry->set_domain_dimensions(&recon_dims);
            Ry->set_codomain_dimensions(&recon_dims);

            boost::shared_ptr<cuPartialDerivativeOperator<float_complext, 4>>
                Rz(new cuPartialDerivativeOperator<float_complext, 4>(2));
            Rz->set_weight(recon_params.lambda_spatial_imoco * (resx * resx) / (resz * resz));
            Rz->set_domain_dimensions(&recon_dims);
            Rz->set_codomain_dimensions(&recon_dims);

            boost::shared_ptr<cuPartialDerivativeOperator<float_complext, 4>>
                Rt(new cuPartialDerivativeOperator<float_complext, 4>(3));

            Rt->set_weight(recon_params.lambda_time);

            Rt->set_domain_dimensions(&recon_dims);
            Rt->set_codomain_dimensions(&recon_dims);
            // TV->set_domain_dimensions(&recon_dims);
            solver_.add_regularization_operator(Rx, recon_params.norm);
            solver_.add_regularization_operator(Ry, recon_params.norm);
            solver_.add_regularization_operator(Rz, recon_params.norm);
            solver_.add_regularization_operator(Rt, recon_params.norm);

            // solver_.add_regularization_group_operator(Rx);
            // solver_.add_regularization_group_operator(Ry);
            // solver_.add_regularization_group_operator(Rz);
            // solver_.add_group(recon_params.norm);

            GDEBUG_STREAM("Data_device:" << data->get_device());
            GDEBUG_STREAM("gpus_input_possible[0]:" << recon_params.selectedDevices[0]<< " [1] if exist" << recon_params.selectedDevices[1]);
            solver_.set_gpus(recon_params.selectedDevices);
            cudaSetDevice(data->get_device());
            reg_image = *solver_.solve(data); 

            cuNDArray<float_complext> images_cropped = this->crop_to_recondims<float_complext>(reg_image); // Recon Size (3D+t)

            // crop<float_complext, 4>(uint64d4(0, 0, (image_dims_[2] - recon_params.ematrixSize.z) / 2, 0),
            //                         uint64d4(image_dims_[0], image_dims_[1], recon_params.ematrixSize.z, reg_image.get_size(3)),
            //                         reg_image,
            //                         images_cropped);

            cudaMemcpy(data->get_data_ptr(), hodata.get_data_ptr(), data->get_number_of_elements() * sizeof(float_complext), cudaMemcpyHostToDevice);
            cudaMemcpy(csm->get_data_ptr(), hocsm.get_data_ptr(), csm->get_number_of_elements() * sizeof(float_complext), cudaMemcpyHostToDevice);

            // // de-prep data
            // for (auto ii = 0; ii < dcw->size(); ii++)
            // {
            //     auto inter_acc = std::accumulate(recon_params.shots_per_time.begin(), recon_params.shots_per_time.begin() + (ii), size_t(0)) * data_dims[0];

            //     for (auto iCHA = 0; iCHA < data->get_size(data->get_number_of_dimensions() - 1); iCHA++)
            //     {
            //         auto dataview = cuNDArray<complext<float>>((*dcw)[ii].get_dimensions(), data->data() + inter_acc + stride * iCHA);
            //         dataview /= (*dcw)[ii];
            //     }
            // }
            return images_cropped;
        }


        cuNDArray<float_complext> noncartesian_reconstruction_5D::reconstructiMOCO(
            cuNDArray<float_complext> *data,
            std::vector<cuNDArray<floatd3>> *traj,
            std::vector<cuNDArray<float>> *dcw,
            boost::shared_ptr<cuNDArray<float_complext>> csm,
            float referencePhase)
        {
            // // nhlbi_toolbox::utils::enable_peeraccess();

            auto tresolved_images = reconstruct(data, traj, dcw, csm,false);
            auto avg_images = *sum(&tresolved_images, tresolved_images.get_number_of_dimensions() - 1);
            avg_images /= float_complext((float)tresolved_images.get_size( tresolved_images.get_number_of_dimensions() - 1), (float)0);
            avg_images.squeeze();
            GDEBUG_STREAM("Average image size: " << avg_images.get_size(0) << "x" << avg_images.get_size(1) << "x" << avg_images.get_size(2) << "no dims" << avg_images.get_number_of_dimensions());
            
            int selectedDevice = nhlbi_toolbox::utils::selectCudaDevice();
            if (!recon_params.selectedDevices.empty()){
                selectedDevice=recon_params.selectedDevices.back();
            }

            cudaSetDevice(selectedDevice);
            auto deformations = register_images_gpu(avg_images, referencePhase);
            auto deformation = std::get<0>(deformations);
            auto inv_deformation = std::get<1>(deformations);


            auto data_dims = *data->get_dimensions();
            auto stride = std::accumulate(data_dims.begin(), data_dims.end() - 1, size_t(1), std::multiplies<size_t>());
            cudaSetDevice(data->get_device());

            // prep data and dcw - doing this data save in memory to prevent data from being affected by recon.
            hoNDArray<float_complext> hodata(*data->get_dimensions());
            cudaMemcpy(hodata.get_data_ptr(), data->get_data_ptr(), data->get_number_of_elements() * sizeof(float_complext), cudaMemcpyDeviceToHost);

            hoNDArray<float_complext> hocsm(*csm->get_dimensions());
            cudaMemcpy(hocsm.get_data_ptr(), csm->get_data_ptr(), csm->get_number_of_elements() * sizeof(float_complext), cudaMemcpyDeviceToHost);

            arma::fvec fbins(1);
            fbins.ones();
            std::vector<cuNDArray<float>> scaled_time_vec;
            // prep data and dcw
            // prep data and dcw - doing this data save in memory to prevent data from being affected by recon.
            
            for (auto ii = 0; ii < dcw->size(); ii++)
            {
                auto inter_acc = std::accumulate(recon_params.shots_per_time.begin(), recon_params.shots_per_time.begin() + (ii), size_t(0)) * data_dims[0];
                auto tmp = cuNDArray<float>(*(*dcw)[ii].get_dimensions());
                fill(&tmp, float(0.0));
                scaled_time_vec.push_back(tmp);
                for (auto iCHA = 0; iCHA < recon_params.numberChannels; iCHA++)
                {
                    auto dataview = cuNDArray<complext<float>>((*dcw)[ii].get_dimensions(), data->data() + inter_acc + stride * iCHA);
                    dataview *= (*dcw)[ii];
                }
            }

            std::vector<size_t> cwdims = {image_dims_[0], image_dims_[1], image_dims_[2], 1};
            cuNDArray<float_complext> padded_cw(cwdims);
            fill(&padded_cw, float_complext(1.0, 0.0));
            // if (fbins.n_elem > 1 && combination_weights->get_size(3) > 1)
            //     padded_cw = pad<float_complext, 4>(uint64d4(image_dims_[0], image_dims_[1], image_dims_[2], fbins.n_elem), combination_weights, float_complext(0));
            // else
            //     padded_cw = pad<float_complext, 3>(uint64d3(image_dims_[0], image_dims_[1], image_dims_[2]), combination_weights, float_complext(0));
            padded_cw.squeeze();

            recon_dims = {image_dims_[0], image_dims_[1], image_dims_[2], recon_params.shots_per_time.get_size(1)};

            // cudaSetDevice(def->get_device());
            // std::vector<cuNDArray<float>> padded_def;
            // std::vector<cuNDArray<float>> padded_invdef;

            // auto defDims = *def->get_dimensions();
            // // if (defDims[defDims.size() - 1] > 1)
            // defDims.pop_back(); // remove time
            // // defDims.pop_back(); // remove dim that gets collapsed also time

            // stride = std::accumulate(defDims.begin(), defDims.end(), 1,
            //                          std::multiplies<size_t>());

            // recon_dims = {image_dims_[0], image_dims_[1], 3, image_dims_[2]};

            // // for (auto ii = 0; ii < def->get_size(4)*def->get_size(5); ii++)
            // for (auto ii = 0; ii < def->get_size(4); ii++)
            // {
            //     auto defView = cuNDArray<float>(defDims, def->data() + stride * ii);
            //     auto intdefView = cuNDArray<float>(defDims, invdef->data() + stride * ii);

            //     padded_def.push_back(nhlbi_toolbox::utils::padDeformations(defView, recon_dims));
            //     padded_invdef.push_back(nhlbi_toolbox::utils::padDeformations(intdefView, recon_dims));
            // }
            // def->clear();
            // invdef->clear();
            cudaSetDevice(data->get_device());

            recon_dims = {image_dims_[0], image_dims_[1], image_dims_[2], recon_params.shots_per_time.get_size(1)};

            auto E_ = boost::shared_ptr<cuNonCartesianMOCOOperator_fc<float, 3>>(new cuNonCartesianMOCOOperator_fc<float, 3>(ConvolutionType::ATOMIC));
            auto D_ = boost::shared_ptr<cuCgPreconditioner<float_complext>>(new cuCgPreconditioner<float_complext>());

            nhlbi_toolbox::cuGpBbSolver<float_complext> solver_;
            // cuSbcCgSolver<float_complext> solver_;

            solver_.set_max_iterations(recon_params.iterations_imoco);
            // solver_.set_max_outer_iterations(recon_params.iterations);
            // solver_.set_max_inner_iterations(recon_params.iterations_inner);
            cuNDArray<float_complext> reg_image(recon_dims);

            E_->set_shots_per_time(recon_params.shots_per_time);
            E_->setup(from_std_vector<size_t, 3>(image_dims_), from_std_vector<size_t, 3>(image_dims_os_), recon_params.kernel_width_);
            E_->set_codomain_dimensions(&data_dims);
            E_->set_domain_dimensions(&recon_dims);
            E_->set_csm(csm);
            E_->set_dcw(*dcw);
            E_->set_recon_params(recon_params);
            E_->set_combination_weights(&padded_cw);
            E_->set_scaled_time(scaled_time_vec);
            E_->set_fbins(fbins);
            if (!recon_params.doMC_iter)
            {
                E_->set_forward_deformation(&deformation);
                E_->set_backward_deformation(&inv_deformation);
            }
            E_->set_doMC_iter(recon_params.doMC_iter);
            E_->set_iteration_count(recon_params.iteration_count_moco);
            E_->set_refPhase(referencePhase);
            E_->set_eligibleGPUs(recon_params.selectedDevices);
            // cudaSetDevice(padded_def[0].get_device());

            // E_->set_forward_deformation(&padded_def);
            // E_->set_backward_deformation(&padded_invdef);
            cudaSetDevice(data->get_device());

            // do everything before preprocess
            E_->preprocess(*traj);

            auto x0 = boost::make_shared<cuNDArray<float_complext>>(cuNDArray<float_complext>(recon_dims));
            E_->mult_MH(data, x0.get());
            solver_.set_encoding_operator(E_);

            solver_.set_tc_tolerance(recon_params.tolerance);
            solver_.set_output_mode(decltype(solver_)::OUTPUT_VERBOSE);
            solver_.set_x0(x0);

            boost::shared_ptr<cuNDArray<float>> _precon_weights;
            boost::shared_ptr<cuNDArray<float_complext>> precon_weights;

            _precon_weights = sum(abs_square(csm.get()).get(), 3);
            recon_dims = {image_dims_[0], image_dims_[1], recon_params.rmatrixSize.z}; // Cropped to size of Recon Matrix

            cuNDArray<float> _precon_weights_cropped = this->crop_to_recondims<float>(*_precon_weights);

            reciprocal_sqrt_inplace(&_precon_weights_cropped);
            // reciprocal_sqrt_inplace(_precon_weights.get());
            precon_weights = boost::make_shared<cuNDArray<float_complext>>(pad<float_complext, 3>(uint64d3(recon_dims[0], recon_dims[1], image_dims_[2]),
                                                                                                  *real_to_complex<float_complext>(&_precon_weights_cropped), float_complext(0)));

            GDEBUG_STREAM("Reseting prerecon weigths from inf to zero");
            auto ho_prerecon = hoNDArray<std::complex<float>>(std::move(*boost::reinterpret_pointer_cast<hoNDArray<std::complex<float>>>((*precon_weights).to_host())));
            auto ho_prereconr = real(ho_prerecon);
            auto ho_prereconi = imag(ho_prerecon);

            std::replace(ho_prereconr.begin(),ho_prereconr.end(),INFINITY,0.0f);
            std::replace(ho_prereconi.begin(),ho_prereconi.end(),INFINITY,0.0f);
            auto ho_prereconri = *real_imag_to_complex<float_complext>(&ho_prereconr,&ho_prereconi);
            precon_weights = boost::make_shared<cuNDArray<float_complext>>(hoNDArray<float_complext>(ho_prereconri));                                                                                                  

            D_->set_weights(precon_weights);
            solver_.set_preconditioner(D_);

            recon_dims = {image_dims_[0], image_dims_[1], image_dims_[2], recon_params.shots_per_time.get_size(1)};

            boost::shared_ptr<cuPartialDerivativeOperator<float_complext, 4>>
                Rx(new cuPartialDerivativeOperator<float_complext, 4>(0));
            Rx->set_weight(recon_params.lambda_spatial);
            Rx->set_domain_dimensions(&recon_dims);
            Rx->set_codomain_dimensions(&recon_dims);

            boost::shared_ptr<cuPartialDerivativeOperator<float_complext, 4>>
                Ry(new cuPartialDerivativeOperator<float_complext, 4>(1));
            Ry->set_weight(recon_params.lambda_spatial);
            Ry->set_domain_dimensions(&recon_dims);
            Ry->set_codomain_dimensions(&recon_dims);

            boost::shared_ptr<cuPartialDerivativeOperator<float_complext, 4>>
                Rz(new cuPartialDerivativeOperator<float_complext, 4>(2));
            Rz->set_weight(recon_params.lambda_spatial * (resx * resx) / (resz * resz));
            Rz->set_domain_dimensions(&recon_dims);
            Rz->set_codomain_dimensions(&recon_dims);

            boost::shared_ptr<cuPartialDerivativeOperator<float_complext, 4>>
                Rt(new cuPartialDerivativeOperator<float_complext, 4>(3));

            Rt->set_weight(recon_params.lambda_time);

            Rt->set_domain_dimensions(&recon_dims);
            Rt->set_codomain_dimensions(&recon_dims);
            // TV->set_domain_dimensions(&recon_dims);
            solver_.add_regularization_operator(Rx, recon_params.norm);
            solver_.add_regularization_operator(Ry, recon_params.norm);
            solver_.add_regularization_operator(Rz, recon_params.norm);
            solver_.add_regularization_operator(Rt, recon_params.norm);

            // solver_.add_regularization_group_operator(Rx);
            // solver_.add_regularization_group_operator(Ry);
            // solver_.add_regularization_group_operator(Rz);
            // solver_.add_group(recon_params.norm);

            GDEBUG_STREAM("Data_device:" << data->get_device());
            GDEBUG_STREAM("gpus_input_possible[0]:" << recon_params.selectedDevices[0]<< " [1] if exist" << recon_params.selectedDevices[1]);
            solver_.set_gpus(recon_params.selectedDevices);
            cudaSetDevice(data->get_device());
            reg_image = *solver_.solve(data); 

            cuNDArray<float_complext> images_cropped = this->crop_to_recondims<float_complext>(reg_image); // Recon Size (3D+t)

            // crop<float_complext, 4>(uint64d4(0, 0, (image_dims_[2] - recon_params.ematrixSize.z) / 2, 0),
            //                         uint64d4(image_dims_[0], image_dims_[1], recon_params.ematrixSize.z, reg_image.get_size(3)),
            //                         reg_image,
            //                         images_cropped);

            cudaMemcpy(data->get_data_ptr(), hodata.get_data_ptr(), data->get_number_of_elements() * sizeof(float_complext), cudaMemcpyHostToDevice);
            cudaMemcpy(csm->get_data_ptr(), hocsm.get_data_ptr(), csm->get_number_of_elements() * sizeof(float_complext), cudaMemcpyHostToDevice);

            // // de-prep data
            // for (auto ii = 0; ii < dcw->size(); ii++)
            // {
            //     auto inter_acc = std::accumulate(recon_params.shots_per_time.begin(), recon_params.shots_per_time.begin() + (ii), size_t(0)) * data_dims[0];

            //     for (auto iCHA = 0; iCHA < data->get_size(data->get_number_of_dimensions() - 1); iCHA++)
            //     {
            //         auto dataview = cuNDArray<complext<float>>((*dcw)[ii].get_dimensions(), data->data() + inter_acc + stride * iCHA);
            //         dataview /= (*dcw)[ii];
            //     }
            // }
            return images_cropped;
        }
    
      cuNDArray<float_complext> noncartesian_reconstruction_5D::reconstructiMOCO_withdef(
            cuNDArray<float_complext> *data,
            std::vector<cuNDArray<floatd3>> *traj,
            std::vector<cuNDArray<float>> *dcw,
            boost::shared_ptr<cuNDArray<float_complext>> csm,
            cuNDArray<float> *def,
            cuNDArray<float> *invdef,
            float referencePhase)
        {
            // // nhlbi_toolbox::utils::enable_peeraccess();

            int selectedDevice = nhlbi_toolbox::utils::selectCudaDevice();
            if (!recon_params.selectedDevices.empty()){
                selectedDevice=recon_params.selectedDevices.back();
            }

            cudaSetDevice(selectedDevice);

            auto data_dims = *data->get_dimensions();
            auto stride = std::accumulate(data_dims.begin(), data_dims.end() - 1, size_t(1), std::multiplies<size_t>());
            cudaSetDevice(data->get_device());

            // prep data and dcw - doing this data save in memory to prevent data from being affected by recon.
            hoNDArray<float_complext> hodata(*data->get_dimensions());
            cudaMemcpy(hodata.get_data_ptr(), data->get_data_ptr(), data->get_number_of_elements() * sizeof(float_complext), cudaMemcpyDeviceToHost);

            hoNDArray<float_complext> hocsm(*csm->get_dimensions());
            cudaMemcpy(hocsm.get_data_ptr(), csm->get_data_ptr(), csm->get_number_of_elements() * sizeof(float_complext), cudaMemcpyDeviceToHost);

            arma::fvec fbins(1);
            fbins.ones();
            std::vector<cuNDArray<float>> scaled_time_vec;
            // prep data and dcw
            // prep data and dcw - doing this data save in memory to prevent data from being affected by recon.
            
            for (auto ii = 0; ii < dcw->size(); ii++)
            {
                auto inter_acc = std::accumulate(recon_params.shots_per_time.begin(), recon_params.shots_per_time.begin() + (ii), size_t(0)) * data_dims[0];
                auto tmp = cuNDArray<float>(*(*dcw)[ii].get_dimensions());
                fill(&tmp, float(0.0));
                scaled_time_vec.push_back(tmp);
                for (auto iCHA = 0; iCHA < recon_params.numberChannels; iCHA++)
                {
                    auto dataview = cuNDArray<complext<float>>((*dcw)[ii].get_dimensions(), data->data() + inter_acc + stride * iCHA);
                    dataview *= (*dcw)[ii];
                }
            }

            std::vector<size_t> cwdims = {image_dims_[0], image_dims_[1], image_dims_[2], 1};
            cuNDArray<float_complext> padded_cw(cwdims);
            fill(&padded_cw, float_complext(1.0, 0.0));
            // if (fbins.n_elem > 1 && combination_weights->get_size(3) > 1)
            //     padded_cw = pad<float_complext, 4>(uint64d4(image_dims_[0], image_dims_[1], image_dims_[2], fbins.n_elem), combination_weights, float_complext(0));
            // else
            //     padded_cw = pad<float_complext, 3>(uint64d3(image_dims_[0], image_dims_[1], image_dims_[2]), combination_weights, float_complext(0));
            padded_cw.squeeze();

            recon_dims = {image_dims_[0], image_dims_[1], image_dims_[2], recon_params.shots_per_time.get_size(1)};

            cudaSetDevice(def->get_device());
            std::vector<cuNDArray<float>> padded_def;
            std::vector<cuNDArray<float>> padded_invdef;

            auto defDims = *def->get_dimensions();
            // if (defDims[defDims.size() - 1] > 1)
            defDims.pop_back(); // remove time
            // defDims.pop_back(); // remove dim that gets collapsed also time

            stride = std::accumulate(defDims.begin(), defDims.end(), 1,
                                     std::multiplies<size_t>());

            recon_dims = {image_dims_[0], image_dims_[1], 3, image_dims_[2]};

            // for (auto ii = 0; ii < def->get_size(4)*def->get_size(5); ii++)
            for (auto ii = 0; ii < def->get_size(4); ii++)
            {
                auto defView = cuNDArray<float>(defDims, def->data() + stride * ii);
                auto intdefView = cuNDArray<float>(defDims, invdef->data() + stride * ii);

                padded_def.push_back(nhlbi_toolbox::utils::padDeformations(defView, recon_dims));
                padded_invdef.push_back(nhlbi_toolbox::utils::padDeformations(intdefView, recon_dims));
            }
            def->clear();
            invdef->clear();
            cudaSetDevice(data->get_device());

            recon_dims = {image_dims_[0], image_dims_[1], image_dims_[2], recon_params.shots_per_time.get_size(1)};

            auto E_ = boost::shared_ptr<cuNonCartesianMOCOOperator_fc<float, 3>>(new cuNonCartesianMOCOOperator_fc<float, 3>(ConvolutionType::ATOMIC));
            auto D_ = boost::shared_ptr<cuCgPreconditioner<float_complext>>(new cuCgPreconditioner<float_complext>());

            nhlbi_toolbox::cuGpBbSolver<float_complext> solver_;
            // cuSbcCgSolver<float_complext> solver_;

            solver_.set_max_iterations(recon_params.iterations_imoco);
            // solver_.set_max_outer_iterations(recon_params.iterations);
            // solver_.set_max_inner_iterations(recon_params.iterations_inner);
            cuNDArray<float_complext> reg_image(recon_dims);

            E_->set_shots_per_time(recon_params.shots_per_time);
            E_->setup(from_std_vector<size_t, 3>(image_dims_), from_std_vector<size_t, 3>(image_dims_os_), recon_params.kernel_width_);
            E_->set_codomain_dimensions(&data_dims);
            E_->set_domain_dimensions(&recon_dims);
            E_->set_csm(csm);
            E_->set_dcw(*dcw);
            E_->set_recon_params(recon_params);
            E_->set_combination_weights(&padded_cw);
            E_->set_scaled_time(scaled_time_vec);
            E_->set_fbins(fbins);
            if (!recon_params.doMC_iter)
            {
                E_->set_forward_deformation(&padded_def);
                E_->set_backward_deformation(&padded_invdef);
            }
            E_->set_doMC_iter(recon_params.doMC_iter);
            E_->set_iteration_count(recon_params.iteration_count_moco);
            E_->set_refPhase(referencePhase);
            E_->set_eligibleGPUs(recon_params.selectedDevices);
            // cudaSetDevice(padded_def[0].get_device());

            // E_->set_forward_deformation(&padded_def);
            // E_->set_backward_deformation(&padded_invdef);
            cudaSetDevice(data->get_device());

            // do everything before preprocess
            E_->preprocess(*traj);

            auto x0 = boost::make_shared<cuNDArray<float_complext>>(cuNDArray<float_complext>(recon_dims));
            E_->mult_MH(data, x0.get());
            solver_.set_encoding_operator(E_);

            solver_.set_tc_tolerance(recon_params.tolerance);
            solver_.set_output_mode(decltype(solver_)::OUTPUT_VERBOSE);
            solver_.set_x0(x0);

            boost::shared_ptr<cuNDArray<float>> _precon_weights;
            boost::shared_ptr<cuNDArray<float_complext>> precon_weights;

            _precon_weights = sum(abs_square(csm.get()).get(), 3);
            recon_dims = {image_dims_[0], image_dims_[1], recon_params.rmatrixSize.z}; // Cropped to size of Recon Matrix

            cuNDArray<float> _precon_weights_cropped = this->crop_to_recondims<float>(*_precon_weights);

            reciprocal_sqrt_inplace(&_precon_weights_cropped);
            // reciprocal_sqrt_inplace(_precon_weights.get());
            precon_weights = boost::make_shared<cuNDArray<float_complext>>(pad<float_complext, 3>(uint64d3(recon_dims[0], recon_dims[1], image_dims_[2]),
                                                                                                  *real_to_complex<float_complext>(&_precon_weights_cropped), float_complext(0)));

            GDEBUG_STREAM("Reseting prerecon weigths from inf to zero");
            auto ho_prerecon = hoNDArray<std::complex<float>>(std::move(*boost::reinterpret_pointer_cast<hoNDArray<std::complex<float>>>((*precon_weights).to_host())));
            auto ho_prereconr = real(ho_prerecon);
            auto ho_prereconi = imag(ho_prerecon);

            std::replace(ho_prereconr.begin(),ho_prereconr.end(),INFINITY,0.0f);
            std::replace(ho_prereconi.begin(),ho_prereconi.end(),INFINITY,0.0f);
            auto ho_prereconri = *real_imag_to_complex<float_complext>(&ho_prereconr,&ho_prereconi);
            precon_weights = boost::make_shared<cuNDArray<float_complext>>(hoNDArray<float_complext>(ho_prereconri));                                                                                                  

            D_->set_weights(precon_weights);
            solver_.set_preconditioner(D_);

            recon_dims = {image_dims_[0], image_dims_[1], image_dims_[2], recon_params.shots_per_time.get_size(1)};

            boost::shared_ptr<cuPartialDerivativeOperator<float_complext, 4>>
                Rx(new cuPartialDerivativeOperator<float_complext, 4>(0));
            Rx->set_weight(recon_params.lambda_spatial);
            Rx->set_domain_dimensions(&recon_dims);
            Rx->set_codomain_dimensions(&recon_dims);

            boost::shared_ptr<cuPartialDerivativeOperator<float_complext, 4>>
                Ry(new cuPartialDerivativeOperator<float_complext, 4>(1));
            Ry->set_weight(recon_params.lambda_spatial);
            Ry->set_domain_dimensions(&recon_dims);
            Ry->set_codomain_dimensions(&recon_dims);

            boost::shared_ptr<cuPartialDerivativeOperator<float_complext, 4>>
                Rz(new cuPartialDerivativeOperator<float_complext, 4>(2));
            Rz->set_weight(recon_params.lambda_spatial * (resx * resx) / (resz * resz));
            Rz->set_domain_dimensions(&recon_dims);
            Rz->set_codomain_dimensions(&recon_dims);

            boost::shared_ptr<cuPartialDerivativeOperator<float_complext, 4>>
                Rt(new cuPartialDerivativeOperator<float_complext, 4>(3));

            Rt->set_weight(recon_params.lambda_time);

            Rt->set_domain_dimensions(&recon_dims);
            Rt->set_codomain_dimensions(&recon_dims);
            // TV->set_domain_dimensions(&recon_dims);
            solver_.add_regularization_operator(Rx, recon_params.norm);
            solver_.add_regularization_operator(Ry, recon_params.norm);
            solver_.add_regularization_operator(Rz, recon_params.norm);
            solver_.add_regularization_operator(Rt, recon_params.norm);

            // solver_.add_regularization_group_operator(Rx);
            // solver_.add_regularization_group_operator(Ry);
            // solver_.add_regularization_group_operator(Rz);
            // solver_.add_group(recon_params.norm);

            GDEBUG_STREAM("Data_device:" << data->get_device());
            GDEBUG_STREAM("gpus_input_possible[0]:" << recon_params.selectedDevices[0]<< " [1] if exist" << recon_params.selectedDevices[1]);
            solver_.set_gpus(recon_params.selectedDevices);
            cudaSetDevice(data->get_device());
            reg_image = *solver_.solve(data); 

            cuNDArray<float_complext> images_cropped = this->crop_to_recondims<float_complext>(reg_image); // Recon Size (3D+t)

            // crop<float_complext, 4>(uint64d4(0, 0, (image_dims_[2] - recon_params.ematrixSize.z) / 2, 0),
            //                         uint64d4(image_dims_[0], image_dims_[1], recon_params.ematrixSize.z, reg_image.get_size(3)),
            //                         reg_image,
            //                         images_cropped);

            cudaMemcpy(data->get_data_ptr(), hodata.get_data_ptr(), data->get_number_of_elements() * sizeof(float_complext), cudaMemcpyHostToDevice);
            cudaMemcpy(csm->get_data_ptr(), hocsm.get_data_ptr(), csm->get_number_of_elements() * sizeof(float_complext), cudaMemcpyHostToDevice);

            // // de-prep data
            // for (auto ii = 0; ii < dcw->size(); ii++)
            // {
            //     auto inter_acc = std::accumulate(recon_params.shots_per_time.begin(), recon_params.shots_per_time.begin() + (ii), size_t(0)) * data_dims[0];

            //     for (auto iCHA = 0; iCHA < data->get_size(data->get_number_of_dimensions() - 1); iCHA++)
            //     {
            //         auto dataview = cuNDArray<complext<float>>((*dcw)[ii].get_dimensions(), data->data() + inter_acc + stride * iCHA);
            //         dataview /= (*dcw)[ii];
            //     }
            // }
            return images_cropped;
        }
    
    
    }



}
