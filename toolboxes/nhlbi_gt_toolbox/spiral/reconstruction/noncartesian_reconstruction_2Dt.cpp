#include "noncartesian_reconstruction_2Dt.h"
#include "util_functions.h"
#include "cuNonCartesianTSenseOperator.h"
#include <cuSbcCgSolver.h>
#include <cuNDArray_elemwise.h>
#include "cuNDArray_elemwise.h"
#include <cuHaarWaveletOperator.h>

using namespace Gadgetron;
namespace nhlbi_toolbox
{
    namespace reconstruction
    {
        
        cuNDArray<float_complext> noncartesian_reconstruction_2Dt::reconstruct_CGSense(
            cuNDArray<float_complext> *data,
            std::vector<cuNDArray<vector_td<float, 2>>> *traj,
            std::vector<cuNDArray<float>> *dcw,
            boost::shared_ptr<cuNDArray<float_complext>> csm)
        {
            auto data_dims = *data->get_dimensions();
            cudaSetDevice(data->get_device());

            auto stride = std::accumulate(data_dims.begin(), data_dims.end() - 1, size_t(1), std::multiplies<size_t>());
            
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

            auto E_ = boost::shared_ptr<cuNonCartesianTSenseOperator<float, 2>>(new cuNonCartesianTSenseOperator<float, 2>(ConvolutionType::ATOMIC));
            auto D_ = boost::shared_ptr<cuCgPreconditioner<float_complext>>(new cuCgPreconditioner<float_complext>());
            //recon_params.shots_per_time.get_size(0): cardiac frames ; recon_params.shots_per_time.get_size(1):set
            recon_dims = {image_dims_[0], image_dims_[1],recon_params.shots_per_time.get_size(0),recon_params.shots_per_time.get_size(1)};
            std::cout << "recon_dims: ";
            for (const auto& dim : recon_dims) {
                std::cout << dim << " ";
            }
            std::cout << std::endl;
            cuGpBbSolver<float_complext> solver_;
            E_->set_recon_params(recon_params);
            cudaSetDevice(data->get_device());

            solver_.set_max_iterations(recon_params.iterations);
            cuNDArray<float_complext> reg_image(recon_dims);
            E_->set_shots_per_time(recon_params.shots_per_time);
            E_->setup(from_std_vector<size_t, 2>(image_dims_), from_std_vector<size_t, 2>(image_dims_os_), recon_params.kernel_width_);
            E_->set_codomain_dimensions(&data_dims);
            E_->set_domain_dimensions(&recon_dims);
            E_->set_csm(csm);
            E_->set_dcw(*dcw);
            E_->preprocess(*traj);

            solver_.set_encoding_operator(E_);
            solver_.set_tc_tolerance(recon_params.tolerance);
            solver_.set_output_mode(decltype(solver_)::OUTPUT_VERBOSE);
            boost::shared_ptr<cuNDArray<float>> _precon_weights;
            _precon_weights = sum(abs_square(csm.get()).get(), 2);
            reciprocal_sqrt_inplace(_precon_weights.get());

            boost::shared_ptr<cuNDArray<float_complext>> precon_weights = boost::make_shared<cuNDArray<float_complext>>(*real_to_complex<float_complext>(_precon_weights.get()));
            D_->set_weights(precon_weights);
            solver_.set_preconditioner(D_);

            boost::shared_ptr<cuPartialDerivativeOperator<float_complext, 4>>
                Rt(new cuPartialDerivativeOperator<float_complext, 4>(2));

            Rt->set_weight(recon_params.lambda_time);
            Rt->set_domain_dimensions(&recon_dims);
            Rt->set_codomain_dimensions(&recon_dims);
            solver_.add_regularization_operator(Rt, recon_params.norm);

            boost::shared_ptr<cuPartialDerivativeOperator<float_complext, 4>>
                Rx(new cuPartialDerivativeOperator<float_complext, 4>(0));
            Rx->set_weight(recon_params.lambda_spatial);
            Rx->set_domain_dimensions(&recon_dims);
            Rx->set_codomain_dimensions(&recon_dims);
            solver_.add_regularization_operator(Rx, recon_params.norm);
            boost::shared_ptr<cuPartialDerivativeOperator<float_complext, 4>>
                Ry(new cuPartialDerivativeOperator<float_complext, 4>(1));
            Ry->set_weight(recon_params.lambda_spatial);
            Ry->set_domain_dimensions(&recon_dims);
            Ry->set_codomain_dimensions(&recon_dims);
            solver_.add_regularization_operator(Ry, recon_params.norm);

            GDEBUG_STREAM("Data_device:" << data->get_device());
            GDEBUG_STREAM("gpus_input_possible[0]:" << recon_params.selectedDevices[0]<< " [1] if exist" << recon_params.selectedDevices[1]);
            solver_.set_gpus(recon_params.selectedDevices);
            reg_image = *solver_.solve(data);

            auto reg_image_dims = *reg_image.get_dimensions();
            std::cout << "reg_image dimensions: ";
            for (const auto& dim : reg_image_dims) {
                std::cout << dim << " ";
            }
            std::cout << std::endl;
            cuNDArray<float_complext> images_cropped = this->crop_to_recondims<float_complext>(reg_image);

        return images_cropped;
        }

        cuNDArray<float_complext> noncartesian_reconstruction_2Dt::reconstruct_CGSense_wav(
            cuNDArray<float_complext> *data,
            std::vector<cuNDArray<vector_td<float, 2>>> *traj,
            std::vector<cuNDArray<float>> *dcw,
            boost::shared_ptr<cuNDArray<float_complext>> csm)
        {
            auto data_dims = *data->get_dimensions();

            auto stride = std::accumulate(data_dims.begin(), data_dims.end() - 1, size_t(1), std::multiplies<size_t>());
            
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

            auto E_ = boost::shared_ptr<cuNonCartesianTSenseOperator<float, 2>>(new cuNonCartesianTSenseOperator<float, 2>(ConvolutionType::ATOMIC));
            auto D_ = boost::shared_ptr<cuCgPreconditioner<float_complext>>(new cuCgPreconditioner<float_complext>());
            //recon_params.shots_per_time.get_size(0): cardiac frames ; recon_params.shots_per_time.get_size(1):set
            recon_dims = {image_dims_[0], image_dims_[1],recon_params.shots_per_time.get_size(0),recon_params.shots_per_time.get_size(1)};
            
            cuGpBbSolver<float_complext> solver_;
            E_->set_recon_params(recon_params);
            solver_.set_max_iterations(recon_params.iterations);
            cuNDArray<float_complext> reg_image(recon_dims);
            E_->set_shots_per_time(recon_params.shots_per_time);
            E_->setup(from_std_vector<size_t, 2>(image_dims_), from_std_vector<size_t, 2>(image_dims_os_), recon_params.kernel_width_);
            E_->set_codomain_dimensions(&data_dims);
            E_->set_domain_dimensions(&recon_dims);
            E_->set_csm(csm);
            E_->set_dcw(*dcw);
            E_->preprocess(*traj);

            solver_.set_encoding_operator(E_);
            solver_.set_tc_tolerance(recon_params.tolerance);
            solver_.set_output_mode(decltype(solver_)::OUTPUT_VERBOSE);
            boost::shared_ptr<cuNDArray<float>> _precon_weights;
            _precon_weights = sum(abs_square(csm.get()).get(), 2);
            reciprocal_sqrt_inplace(_precon_weights.get());

            boost::shared_ptr<cuNDArray<float_complext>> precon_weights = boost::make_shared<cuNDArray<float_complext>>(*real_to_complex<float_complext>(_precon_weights.get()));
            D_->set_weights(precon_weights);
            solver_.set_preconditioner(D_);

            boost::shared_ptr<cuPartialDerivativeOperator<float_complext, 4>>
                Rt(new cuPartialDerivativeOperator<float_complext, 4>(2));

            Rt->set_weight(recon_params.lambda_time);
            Rt->set_domain_dimensions(&recon_dims);
            Rt->set_codomain_dimensions(&recon_dims);
            solver_.add_regularization_operator(Rt, recon_params.norm);

            boost::shared_ptr<cuHaarWaveletOperator<float_complext, 2>>
                Rx(new cuHaarWaveletOperator<float_complext, 2>);

            
            Rx->set_weight(recon_params.lambda_spatial);
            Rx->set_domain_dimensions(&recon_dims);
            Rx->set_codomain_dimensions(&recon_dims);
            solver_.add_regularization_operator(Rx, recon_params.norm);
            

            GDEBUG_STREAM("Data_device:" << data->get_device());
            GDEBUG_STREAM("gpus_input_possible[0]:" << recon_params.selectedDevices[0]<< " [1] if exist" << recon_params.selectedDevices[1]);
            solver_.set_gpus(recon_params.selectedDevices);
            reg_image = *solver_.solve(data);

            cuNDArray<float_complext> images_cropped = this->crop_to_recondims<float_complext>(reg_image);


        return images_cropped;
        }

        std::tuple<cuNDArray<float_complext>,
                   std::vector<cuNDArray<vector_td<float, 2>>>,
                   std::vector<cuNDArray<float>>>
        noncartesian_reconstruction_2Dt::organize_data(
            hoNDArray<float_complext> *data,
            hoNDArray<vector_td<float, 2>> *traj,
            hoNDArray<float> *dcw)
        {
            GDEBUG_STREAM("Deprecated function !");
            auto totalnumInt = std::accumulate(recon_params.shots_per_time.begin(), recon_params.shots_per_time.end(), size_t(0));
            std::vector<size_t> data_dims = {data->get_size(0), totalnumInt, recon_params.numberChannels};

            auto cuData = boost::make_shared<cuNDArray<float_complext>>(cuNDArray<float_complext>(data_dims));
            std::vector<cuNDArray<vector_td<float, 2>>> cuTrajVec;
            std::vector<cuNDArray<float>> cuDCWVec;

            auto stride = std::accumulate(data_dims.begin(), data_dims.end() - 1, size_t(1), std::multiplies<size_t>());

            auto cutraj = cuNDArray<vector_td<float, 2>>(traj);
            auto cudcw = cuNDArray<float>(dcw);

            for (auto ii = 0; ii < recon_params.shots_per_time.get_number_of_elements(); ii++)
            {
                auto inter_acc = std::accumulate(recon_params.shots_per_time.begin(), recon_params.shots_per_time.begin() + (ii), size_t(0)) * data_dims[0];

                
                std::vector<size_t> flat_dims = {data_dims[0] * *(recon_params.shots_per_time.begin() + ii)};
                cuNDArray<vector_td<float, 2>> flat_traj(flat_dims, cutraj.get_data_ptr() + inter_acc);
                auto cu_dcf = estimate_dcf(&flat_traj);
                cuTrajVec.push_back(flat_traj);
                cuDCWVec.push_back(cu_dcf);
                if (recon_params.numberChannels == data->get_size(1)) // if the data is not permuted to and is of the shape RO CHA INT then use this code
                {
                    for (size_t iCHA = 0; iCHA < recon_params.numberChannels; iCHA++)
                    {
                        for (auto jj = 0; jj < *(recon_params.shots_per_time.begin() + ii); jj++)
                        {
                            cudaMemcpy(cuData.get()->get_data_ptr() + inter_acc + stride * iCHA + data_dims[0] * jj,
                                       data->get_data_ptr() + data->get_size(0) * iCHA + data->get_size(0) * data->get_size(1) * (jj) + inter_acc,
                                       data->get_size(0) * sizeof(complext<float>), cudaMemcpyDefault);
                        }
                    }
                }
                else
                { // if the data is permuted to be RO INT CHA then use this code
                    for (size_t iCHA = 0; iCHA < recon_params.numberChannels; iCHA++)
                    {
                        for (auto jj = 0; jj < *(recon_params.shots_per_time.begin() + ii); jj++)
                        {
                            cudaMemcpy(cuData.get()->get_data_ptr() + inter_acc + stride * iCHA + data_dims[0] * jj,
                                       data->get_data_ptr() + data->get_size(0) * (jj) + inter_acc + data->get_size(0) * data->get_size(1) * iCHA,
                                       data->get_size(0) * sizeof(complext<float>), cudaMemcpyDefault);
                        }
                    }
                }
            }
            return std::make_tuple(std::move(*cuData), std::move(cuTrajVec), std::move(cuDCWVec));
        }



    }
}