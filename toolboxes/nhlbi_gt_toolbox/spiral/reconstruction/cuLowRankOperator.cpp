#include "cuLowRankOperator.h"
#include <cuNDArray_operators.h>
#include <cuNDArray_elemwise.h>
#include <cuNDArray_blas.h>
#include <cuNDArray_utils.h>
#include <vector_td_utilities.h>
#include <vector_td_io.h>
#include <stdio.h>
#include <iostream>
#include <numeric>
#include "gpuSVD.cuh"
using namespace Gadgetron;

template< unsigned int D> void cuLowRankOperator<D>::mult_M(cuNDArray<float_complext>* in, cuNDArray<float_complext>* out, bool accumulate ){
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


template< unsigned int D> void cuLowRankOperator<D>::mult_MH(cuNDArray<float_complext>* in, cuNDArray<float_complext>* out, bool accumulate ){


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

template< unsigned int D> void cuLowRankOperator<D>::mult_MH_M(cuNDArray<float_complext>* in, cuNDArray<float_complext>* out, bool accumulate ){

	if( !in || !out || in->get_number_of_elements() != out->get_number_of_elements() ){
      throw std::runtime_error( "Low Rank : array dimensions mismatch.");

    }
    GDEBUG_STREAM("low_rank_mult_MH_M");
    GadgetronTimer timer("Low Rank:"); 
    hoNDArray<std::complex<float>> ho_in = hoNDArray<std::complex<float>>(std::move(*boost::reinterpret_pointer_cast<hoNDArray<std::complex<float>>>((*in).to_host())));

    GDEBUG_STREAM("Ho_in ND" << ho_in.get_number_of_dimensions() << " X " << ho_in.get_size(0) << " Y " << ho_in.get_size(1)  << " Z " << ho_in.get_size(2) << " Cardiac " << ho_in.get_size(3) << " Nele " <<ho_in.get_number_of_elements());
             
    
    hoNDArray<std::complex<float>> ho_out;
    //REAL lamda = this->get_weight();
    float lamda =float(this->get_weight());
    float alpha_LR = this->get_alphaLR();
    GDEBUG_STREAM("Paramaters weights " << lamda<< " alpha_LR" <<alpha_LR );

    try
    {
        PythonFunction<hoNDArray<std::complex<float>>> low_rank_mult_MH_M("low_rank", "mult_M");
        GDEBUG_STREAM("low_rank_mult_MH_M start");
        ho_out = low_rank_mult_MH_M(ho_in,D,lamda,alpha_LR);
        GDEBUG_STREAM("low_rank_mult_MH_M end ");
    }
    catch (...)
    {
        GERROR_STREAM("Something broke");
        ho_out.create(out->get_number_of_elements());
        ho_out.fill(1);
    }
    cudaMemcpy(out->get_data_ptr(), ho_out.get_data_ptr(), out->get_number_of_elements() * sizeof(float_complext), cudaMemcpyHostToDevice);
            

}

template< unsigned int D> void cuLowRankOperator<D>::prox(cuNDArray<float_complext>* in, cuNDArray<float_complext>* out, bool accumulate ){

	if( !in || !out || in->get_number_of_elements() != out->get_number_of_elements() ){
      throw std::runtime_error( "Low Rank : array dimensions mismatch.");

    }
    GDEBUG_STREAM("PROX Low rank");
    GadgetronTimer timer("Low Rank:");
    
    float lamda =float(this->get_weight());
    float alpha_LR = this->get_alphaLR();
    
    gpuSVD svd;
    std::vector<size_t> dim_initial={in->get_size(0),in->get_size(1),in->get_size(2), in->get_size(3)};
    
    std::vector<size_t> dim_svd={in->get_size(0)*in->get_size(1)*in->get_size(2), in->get_size(3)};
    GDEBUG_STREAM("dim_svd: " << dim_svd[0] << " " << dim_svd[1] << " " << in->get_size(0) << " " << in->get_size(1) << " " << in->get_size(2) << " " << in->get_size(3));
    in->reshape(dim_svd);
    auto R=svd.batch_LR(in, 1,alpha_LR);
    R.reshape(out->get_dimensions());
    cudaMemcpy(out->get_data_ptr(), R.get_data_ptr(), R.get_number_of_elements() * sizeof(float_complext), cudaMemcpyDefault);
    in->reshape(dim_initial);
    /*
    hoNDArray<std::complex<float>> ho_in = hoNDArray<std::complex<float>>(std::move(*boost::reinterpret_pointer_cast<hoNDArray<std::complex<float>>>((*in).to_host())));

    GDEBUG_STREAM("Ho_in ND" << ho_in.get_number_of_dimensions() << " X " << ho_in.get_size(0) << " Y " << ho_in.get_size(1)  << " Z " << ho_in.get_size(2) << " Cardiac " << ho_in.get_size(3) << " Nele " <<ho_in.get_number_of_elements());
             
    
    hoNDArray<std::complex<float>> ho_out;
    //REAL lamda = this->get_weight();
    float lamda =float(this->get_weight());
    float alpha_LR = this->get_alphaLR();
    GDEBUG_STREAM("Paramaters weights " << lamda<< " alpha_LR" <<alpha_LR );
    
    try
    {
        PythonFunction<hoNDArray<std::complex<float>>> low_rank_mult_MH_M("low_rank", "prox_LR");
        GDEBUG_STREAM("prox_LR start");
        ho_out = low_rank_mult_MH_M(ho_in,D,lamda,alpha_LR);
        GDEBUG_STREAM("prox_LR end ");
    }
    catch (...)
    {
        GERROR_STREAM("Something broke");
        ho_out.create(out->get_number_of_elements());
        ho_out.fill(1);
    }
    
    cudaMemcpy(out->get_data_ptr(), ho_out.get_data_ptr(), out->get_number_of_elements() * sizeof(float_complext), cudaMemcpyHostToDevice);
    */     

}


template< unsigned int D> void cuLowRankOperator<D>::gradient(cuNDArray<float_complext>*in, cuNDArray<float_complext>*out, bool accumulate){
    if( in == 0x0 || out == 0x0 ){
        throw std::runtime_error("lowRankOperator::gradient(): Invalid input and/or output array");
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

template class  cuLowRankOperator<1>;
template class  cuLowRankOperator<2>;
template class  cuLowRankOperator<3>;
template class  cuLowRankOperator<4>;
template class  cuLowRankOperator<5>;