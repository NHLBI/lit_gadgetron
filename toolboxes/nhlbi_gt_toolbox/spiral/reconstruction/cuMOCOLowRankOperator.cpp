#include "cuMOCOLowRankOperator.h"
#include <cuNDArray_operators.h>
#include <cuNDArray_elemwise.h>
#include <cuNDArray_blas.h>
#include <cuNDArray_utils.h>
#include <vector_td_utilities.h>
#include <vector_td_io.h>
#include <stdio.h>
#include <iostream>
#include <numeric>
#include "gpuRegistration.cuh"
#include "gpuSVD.cuh"
using namespace Gadgetron;

template< unsigned int D> void cuMOCOLowRankOperator<D>::mult_M(cuNDArray<float_complext>* in, cuNDArray<float_complext>* out, bool accumulate ){
	if( !in || !out || in->get_number_of_elements() != out->get_number_of_elements() ){
      throw std::runtime_error( "Low Rank : array dimensions mismatch.");

    }
    GDEBUG_STREAM("low_rank_mult_M");
    GadgetronTimer timer("Low Rank:"); 
    hoNDArray<std::complex<float>> ho_in = hoNDArray<std::complex<float>>(std::move(*boost::reinterpret_pointer_cast<hoNDArray<std::complex<float>>>((*in).to_host())));
            
    GDEBUG_STREAM("Ho_in ND" << ho_in.get_number_of_dimensions() << " X " << ho_in.get_size(0) << " Y " << ho_in.get_size(1)  << " Z " << ho_in.get_size(1) << " Cardiac " << ho_in.get_size(1) << " Nele " <<ho_in.get_number_of_elements());
            
    hoNDArray<std::complex<float>> ho_out;
    //REAL lamda = this->get_weight();
    float lamda =1;
    float alpha_LR = this->get_alphaLR();
    GDEBUG_STREAM("Paramaters weights " << lamda<< " alpha_LR" <<alpha_LR );
    try
    {
        PythonFunction<hoNDArray<std::complex<float>>> low_rank_mult_M("low_rank", "mult_M");
        GDEBUG_STREAM("low_rank_mult_M start");
        ho_out = low_rank_mult_M(ho_in,D,lamda,alpha_LR);
        GDEBUG_STREAM("low_rank_mult_M end ");
    }
    catch (...)
    {
        GERROR_STREAM("Something broke");
        ho_out.create(out->get_number_of_elements());
        ho_out.fill(1);
    }

	



}


template< unsigned int D> void cuMOCOLowRankOperator<D>::mult_MH(cuNDArray<float_complext>* in, cuNDArray<float_complext>* out, bool accumulate ){


	if( !in || !out || in->get_number_of_elements() != out->get_number_of_elements() ){
      throw std::runtime_error( "Low Rank : array dimensions mismatch.");

    }
    GDEBUG_STREAM("low_rank_mult_MH");
    GadgetronTimer timer("Low Rank:"); 
    hoNDArray<std::complex<float>> ho_in = hoNDArray<std::complex<float>>(std::move(*boost::reinterpret_pointer_cast<hoNDArray<std::complex<float>>>((*in).to_host())));
            
    GDEBUG_STREAM("Ho_in ND" << ho_in.get_number_of_dimensions() << " X " << ho_in.get_size(0) << " Y " << ho_in.get_size(1)  << " Z " << ho_in.get_size(1) << " Cardiac " << ho_in.get_size(1) << " Nele " <<ho_in.get_number_of_elements());
            
    hoNDArray<std::complex<float>> ho_out;
    //REAL lamda = this->get_weight();
    float lamda =1;
    float alpha_LR = this->get_alphaLR();
    GDEBUG_STREAM("Paramaters weights " << lamda<< " alpha_LR" <<alpha_LR );
    try
    {
        PythonFunction<hoNDArray<std::complex<float>>> low_rank_mult_MH("low_rank", "mult_MH");
        GDEBUG_STREAM("low_rank_mult_MH start");
        ho_out = low_rank_mult_MH(ho_in,D,lamda,alpha_LR);
        GDEBUG_STREAM("low_rank_mult_MH  end ");
    }
    catch (...)
    {
        GERROR_STREAM("Something broke");
        ho_out.create(out->get_number_of_elements());
        ho_out.fill(1);
    }
    cudaMemcpy(out->get_data_ptr(), ho_out.get_data_ptr(), out->get_number_of_elements() * sizeof(float_complext), cudaMemcpyHostToDevice);
      
}

template< unsigned int D> void cuMOCOLowRankOperator<D>::mult_MH_M(cuNDArray<float_complext>* in, cuNDArray<float_complext>* out, bool accumulate ){

	if( !in || !out || in->get_number_of_elements() != out->get_number_of_elements() ){
      throw std::runtime_error( "Low Rank : array dimensions mismatch.");

    }
        GDEBUG_STREAM("MOCO low_rank_mult_MH_M");
    GadgetronTimer timer("MOCO Low Rank:"); 
    float refP =this->get_refPhase();
    auto deformations = register_images_gpu(*in, refP);
    auto deformation = std::get<0>(deformations);
    auto inv_deformation = std::get<1>(deformations);

    hoNDArray<std::complex<float>> ho_in = hoNDArray<std::complex<float>>(std::move(*boost::reinterpret_pointer_cast<hoNDArray<std::complex<float>>>((*in).to_host())));



    GDEBUG_STREAM("Ho_in ND" << ho_in.get_number_of_dimensions() << " X " << ho_in.get_size(0) << " Y " << ho_in.get_size(1)  << " Z " << ho_in.get_size(2) << " Cardiac " << ho_in.get_size(3) << " Nele " <<ho_in.get_number_of_elements());


    auto registered_images = applyDeformations(ho_in, deformation);         
    
    hoNDArray<std::complex<float>> registered_images_LR;
    //REAL lamda = this->get_weight();
    float lamda =float(this->get_weight());
    float alpha_LR = this->get_alphaLR();
    GDEBUG_STREAM("Paramaters weights " << lamda<< " alpha_LR" <<alpha_LR );

    try
    {
        PythonFunction<hoNDArray<std::complex<float>>> low_rank_mult_MH_M("low_rank", "mult_M");
        GDEBUG_STREAM("low_rank_mult_MH_M start");
        registered_images_LR = low_rank_mult_MH_M(registered_images,D,lamda,alpha_LR);
        GDEBUG_STREAM("low_rank_mult_MH_M end ");
    }
    catch (...)
    {
        GERROR_STREAM("Something broke");
        registered_images_LR.create(out->get_number_of_elements());
        registered_images_LR.fill(1);
    }

    auto ho_out = applyDeformations(registered_images_LR, inv_deformation);  

    cudaMemcpy(out->get_data_ptr(), ho_out.get_data_ptr(), out->get_number_of_elements() * sizeof(float_complext), cudaMemcpyHostToDevice);
            

}

template< unsigned int D> void cuMOCOLowRankOperator<D>::prox(cuNDArray<float_complext>* in, cuNDArray<float_complext>* out, bool accumulate ){

	if( !in || !out || in->get_number_of_elements() != out->get_number_of_elements() ){
        GDEBUG_STREAM("in num "<< in->get_number_of_elements() << " " <<out->get_number_of_elements() );
      throw std::runtime_error( "Low Rank : array dimensions mismatch.");

    }
    GDEBUG_STREAM("MOCO prox_LR");
    GadgetronTimer timer("MOCO prox_LR:");
    if (debug_){
        nhlbi_toolbox::utils::write_gpu_nd_array<float_complext>(*in, std::string("/opt/data/daudepv/tmp")+ std::string("in_ite_")+std::to_string(this->current_ite)+std::string(".complex"));
        }
    std::vector<cuNDArray<float>> deformation;
    std::vector<cuNDArray<float>> inv_deformation;
    float refP =this->get_refPhase();
    GDEBUG_STREAM("Current iteration " << this->current_ite << " max ite " << this->max_ite << " use_saved_registration " << this->use_saved_registration_ << " max_ite_registration " << this->max_ite_registration);
    if (this->use_saved_registration_ && this->current_ite > this->max_ite_registration) {
        GDEBUG_STREAM("Using saved registration");
        deformation = this->forward_deformation_;
        inv_deformation = this->backward_deformation_;
    } else {
        GDEBUG_STREAM("Using new registration");
        auto deformations = register_images_gpu(*in, refP);
        deformation = std::get<0>(deformations);
        inv_deformation = std::get<1>(deformations);
        if (debug_){
            for (size_t d = 0; d < deformation.size(); d++)
                nhlbi_toolbox::utils::write_gpu_nd_array<float>(deformation[d], std::string("/opt/data/daudepv/tmp")+ std::string("deformation_ite_")+std::to_string(this->current_ite)+std::string("_Nphases_")+std::to_string(d)+std::string(".real"));
        }
    }
    

    hoNDArray<std::complex<float>> ho_in = hoNDArray<std::complex<float>>(std::move(*boost::reinterpret_pointer_cast<hoNDArray<std::complex<float>>>((*in).to_host())));



    GDEBUG_STREAM("Ho_in ND" << ho_in.get_number_of_dimensions() << " X " << ho_in.get_size(0) << " Y " << ho_in.get_size(1)  << " Z " << ho_in.get_size(2) << " Cardiac " << ho_in.get_size(3) << " Nele " <<ho_in.get_number_of_elements() << " REF INDEX " <<refP );


    auto registered_images = applyDeformations(ho_in, deformation);         
    
    //hoNDArray<std::complex<float>> registered_images_LR;
    //REAL lamda = this->get_weight();
    float lamda =float(this->get_weight());
    float alpha_LR = this->get_alphaLR();
    GDEBUG_STREAM("Paramaters weights " << lamda<< " alpha_LR" <<alpha_LR );
    
    gpuSVD svd;
    std::vector<size_t> dim_svd={registered_images.get_size(0)*registered_images.get_size(1)*registered_images.get_size(2), registered_images.get_size(3)};
    GDEBUG_STREAM("dim_svd: " << dim_svd[0] << " " << dim_svd[1] << " " << registered_images.get_size(0) << " " << registered_images.get_size(1) << " " << registered_images.get_size(2) << " " << registered_images.get_size(3));
    cuNDArray<float_complext> cu_rimages(dim_svd);
    cudaMemcpy(cu_rimages.get_data_ptr(), registered_images.get_data_ptr(), cu_rimages.get_number_of_elements() * sizeof(float_complext), cudaMemcpyHostToDevice);
    cuNDArray<float_complext> R=svd.batch_LR(&(cu_rimages),1,alpha_LR);
    std::vector<size_t> dim_R ={registered_images.get_size(0), registered_images.get_size(1), registered_images.get_size(2),registered_images.get_size(3)};
    R.reshape(dim_R);
    auto registered_images_LR = hoNDArray<std::complex<float>>(std::move(*boost::reinterpret_pointer_cast<hoNDArray<std::complex<float>>>(R.to_host())));

    auto ho_out = applyDeformations(registered_images_LR, inv_deformation);  

    cudaMemcpy(out->get_data_ptr(), ho_out.get_data_ptr(), out->get_number_of_elements() * sizeof(float_complext), cudaMemcpyHostToDevice);
    
    if (this->use_saved_registration_ && this->current_ite == this->max_ite_registration) {
        GDEBUG_STREAM("Saving registration");
        this->forward_deformation_ = deformation;
        this->backward_deformation_ = inv_deformation;
    }
    if (!this->use_saved_registration_ ||(this->use_saved_registration_ && this->current_ite < this->max_ite_registration)){
        GDEBUG_STREAM("Clearing memory");
        inv_deformation.clear();
        deformation.clear();
        PythonFunction<> clear_cuda_cache("utils_function", "clear_cuda_cache");
        clear_cuda_cache(gpus);
        GDEBUG_STREAM("CLEARed");
    }
    
    if( this->current_ite == this->max_ite && this->use_saved_registration_) {
         GDEBUG_STREAM("Clearing forward and backward deformation");
       this->forward_deformation_.clear();
       this->backward_deformation_.clear();
    }
    
    this->current_ite++;
    if (debug_){
        nhlbi_toolbox::utils::write_gpu_nd_array<float_complext>(*out, std::string("/opt/data/daudepv/tmp")+ std::string("out_ite_")+std::to_string(this->current_ite)+std::string(".complex"));
    }
    
}


template< unsigned int D> void cuMOCOLowRankOperator<D>::gradient(cuNDArray<float_complext>*in, cuNDArray<float_complext>*out, bool accumulate){
    if( in == 0x0 || out == 0x0 ){
        throw std::runtime_error("MOCOLowRankOperator::gradient(): Invalid input and/or output array");
    }

    cuNDArray<float_complext>* tmp = out;
    if (accumulate){
        tmp = new cuNDArray<float_complext>(out->dimensions());
    }
    if (this->prox_){
        GDEBUG_STREAM("Prox operator ");
        prox(in,tmp,false);
    }else{
        GDEBUG_STREAM("Gradient operator ");
        mult_MH_M(in,tmp,false);
        *tmp *= this->weight_;
    }
     
    if (accumulate){
	    *out += *tmp;
	    delete tmp;
    }
    /*
	hoNDArray<std::complex<float>> ho_in = hoNDArray<std::complex<float>>(std::move(*boost::reinterpret_pointer_cast<hoNDArray<std::complex<float>>>((*in).to_host())));
    
    hoNDArray<std::complex<float>> ho_out;
    GDEBUG_STREAM("Ho_in ND" << ho_in.get_number_of_dimensions() << " X " << ho_in.get_size(0) << " Y " << ho_in.get_size(1)  << " Z " << ho_in.get_size(2) << " Cardiac " << ho_in.get_size(3) << " Nele " <<ho_in.get_number_of_elements());
    GDEBUG_STREAM("Ho_in ND" << out->get_number_of_dimensions() << " X " << out->get_size(0) << " Y " << out->get_size(1)  << " Z " << out->get_size(2) << " Cardiac " << out->get_size(3) << " Nele " <<out->get_number_of_elements());

    



    float lamda =1;
    float alpha_LR = this->get_alphaLR();
    GDEBUG_STREAM("Paramaters weights " << lamda<< " alpha_LR" <<alpha_LR );
    try
    {
        PythonFunction<hoNDArray<std::complex<float>>> low_rank_gradient("low_rank", "gradient");
        GDEBUG_STREAM("low_rank_gradient start");
        ho_out = low_rank_gradient(ho_in,D,lamda,alpha_LR);
        GDEBUG_STREAM("low_rank_gradient end ");
    }
    catch (...)
    {
        GERROR_STREAM("Something broke");
        ho_out.create(out->get_number_of_elements());
        ho_out.fill(1);
    }
    */

}

template< unsigned int D> std::tuple<std::vector<cuNDArray<float>>, std::vector<cuNDArray<float>>> cuMOCOLowRankOperator<D>::register_images_gpu_image_batch(cuNDArray<float_complext> images_gpu, float referencePhase)
        {
            std::vector<cuNDArray<float>> forward_deformation_;
            std::vector<cuNDArray<float>> backward_deformation_;

            PythonFunction<hoNDArray<float>> register_one_image("registration_gadget_call", "registration_one_image");
            auto images_all = hoNDArray<std::complex<float>>(std::move(*boost::reinterpret_pointer_cast<hoNDArray<std::complex<float>>>(images_gpu.to_host())));
            auto image_dim= *(images_all.get_dimensions().get());
            image_dim.pop_back(); // remove time
            auto ndim=images_all.get_number_of_dimensions();
            auto n_images= images_all.get_size(ndim-1);
            
            auto stride = std::accumulate(image_dim.begin(), image_dim.end(), 1,std::multiplies<size_t>()); // product of X,Y,and Z
            auto ref_index =std::min(size_t(float(n_images)*referencePhase),n_images-1);
            GDEBUG_STREAM("All info" << stride << " index " << ref_index << " ndim" << ndim << " nimages" << n_images << " ref_index " << ref_index);
            auto ref_image_view = hoNDArray<std::complex<float>> (image_dim, images_all.data()+stride*ref_index); //bidirectional (n_images/2)
            std::vector<size_t> recon_dims = {images_all.get_size(0), images_all.get_size(1), 3, images_all.get_size(2)};
            for (auto ii = 0; ii < n_images; ii++)
            {
                auto mov_image = hoNDArray<std::complex<float>>(image_dim, images_all.data()+stride*ii);
                auto def = register_one_image(abs(mov_image),abs(ref_image_view),gpus);
                auto defView = cuNDArray<float>(def);
                auto intdefView = defView;
                intdefView *= -1.0f;
                forward_deformation_.push_back(nhlbi_toolbox::utils::padDeformations(defView, recon_dims));
                backward_deformation_.push_back(nhlbi_toolbox::utils::padDeformations(intdefView, recon_dims));
            }
            return std::make_tuple(forward_deformation_, backward_deformation_);
        }

template< unsigned int D> std::tuple<std::vector<cuNDArray<float>>, std::vector<cuNDArray<float>>> cuMOCOLowRankOperator<D>::register_images_gpu(cuNDArray<float_complext> images_gpu, float referencePhase)
        {
            std::vector<cuNDArray<float>> forward_deformation_;
            std::vector<cuNDArray<float>> backward_deformation_;

            PythonFunction<hoNDArray<float>> register_images("registration_gadget_call", "registration_images");
            auto images_all = hoNDArray<std::complex<float>>(std::move(*boost::reinterpret_pointer_cast<hoNDArray<std::complex<float>>>(images_gpu.to_host())));
            auto def = register_images(abs(images_all),referencePhase,gpus);
            auto deformation = cuNDArray<float>(def);
            auto inv_deformation = deformation;
            inv_deformation *= -1.0f;

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

                forward_deformation_.push_back(nhlbi_toolbox::utils::padDeformations(defView, recon_dims));
                backward_deformation_.push_back(nhlbi_toolbox::utils::padDeformations(intdefView, recon_dims));
            }

            return std::make_tuple(forward_deformation_, backward_deformation_);
        }

template< unsigned int D> hoNDArray<std::complex<float>> cuMOCOLowRankOperator<D>::applyDeformations(hoNDArray<std::complex<float>> images_all, std::vector<cuNDArray<float>> deformations)
        {
            gpuRegistration gr;

            auto inputDims = *images_all.get_dimensions();
            auto timages = images_all;
                        if (inputDims.size() > 4)
                timages = permute(images_all, {0, 1, 2, 4, 3});

            timages.reshape({long(timages.get_size(0)), long(timages.get_size(1)), long(timages.get_size(2)), -1});
                        auto registered_images = timages;

            for (auto ii = 0; ii < images_all.get_size(3); ii++)
            {
                auto timage = cuNDArray<float_complext>(hoNDArray<float_complext>(hoNDArray<std::complex<float>>(timages(slice, slice, slice, ii))));
                auto rimage = gr.deform_image(&timage, deformations[ii]);
                registered_images(slice, slice, slice, ii) = hoNDArray<std::complex<float>>(std::move(*boost::reinterpret_pointer_cast<hoNDArray<std::complex<float>>>(rimage.to_host())));
            }
            if (inputDims.size() > 4){
                GDEBUG_STREAM("Code to verify !!! It may not work")
                timages = permute(registered_images, {0, 1, 2, 4, 3});
                timages.reshape(inputDims);
            }else{
                timages=registered_images;
            }
            
            return timages;
        }

template class  cuMOCOLowRankOperator<1>;
template class  cuMOCOLowRankOperator<2>;
template class  cuMOCOLowRankOperator<3>;
template class  cuMOCOLowRankOperator<4>;
template class  cuMOCOLowRankOperator<5>;