#include "cuLocallyLowRankOperator.h"
#include <cuNDArray_operators.h>
#include <cuNDArray_elemwise.h>
#include <cuNDArray_blas.h>
#include <cuNDArray_utils.h>
#include <vector_td_utilities.h>
#include <vector_td_io.h>
#include <stdio.h>
#include <iostream>
#include <numeric>
using namespace Gadgetron;

template< unsigned int D> void cuLocallyLowRankOperator<D>::mult_M(cuNDArray<float_complext>* in, cuNDArray<float_complext>* out, bool accumulate ){
	if( !in || !out || in->get_number_of_elements() != out->get_number_of_elements() ){
      throw std::runtime_error( "Locally Low Rank : array dimensions mismatch.");

    }
   GERROR("mult_M can't be computed");
}


template< unsigned int D> void cuLocallyLowRankOperator<D>::mult_MH(cuNDArray<float_complext>* in, cuNDArray<float_complext>* out, bool accumulate ){


	if( !in || !out || in->get_number_of_elements() != out->get_number_of_elements() ){
      throw std::runtime_error( "Locally Low Rank : array dimensions mismatch.");

    }
    GERROR("mult_MH can't be computed");
}

template< unsigned int D> void cuLocallyLowRankOperator<D>::mult_MH_M(cuNDArray<float_complext>* in, cuNDArray<float_complext>* out, bool accumulate ){

	if( !in || !out || in->get_number_of_elements() != out->get_number_of_elements() ){
      throw std::runtime_error( "Locally Low Rank : array dimensions mismatch.");

    }
    GERROR("Mult_MH_M can't be computed");
}

template< unsigned int D> void cuLocallyLowRankOperator<D>::prox(cuNDArray<float_complext>* in, cuNDArray<float_complext>* out, bool accumulate ){

	if( !in || !out || in->get_number_of_elements() != out->get_number_of_elements() ){
      throw std::runtime_error( "Locally Low Rank : array dimensions mismatch.");

    }
    GDEBUG_STREAM("PROXIMAL Locally Low rank");
    GadgetronTimer timer(" Locally Low Rank:"); 
    hoNDArray<std::complex<float>> ho_in = hoNDArray<std::complex<float>>(std::move(*boost::reinterpret_pointer_cast<hoNDArray<std::complex<float>>>((*in).to_host())));

    GDEBUG_STREAM("Ho_in ND" << ho_in.get_number_of_dimensions() << " X " << ho_in.get_size(0) << " Y " << ho_in.get_size(1)  << " Z " << ho_in.get_size(2) << " Cardiac " << ho_in.get_size(3) << " Nele " <<ho_in.get_number_of_elements());
             
    
    hoNDArray<std::complex<float>> ho_out;
    //REAL lamda = this->get_weight();
    float lamda =float(this->get_weight());
    float alpha_LR = this->get_alphaLR();
    size_t blockLR = this->get_blockLR();
    GDEBUG_STREAM("Paramaters weights " << lamda<< " alpha_LR" <<alpha_LR << blockLR);

    try
    {
        PythonFunction<hoNDArray<std::complex<float>>> prox_LLR_py("low_rank", "prox_LLR");
        GDEBUG_STREAM("prox_LLR start");
        ho_out = prox_LLR_py(ho_in,D,alpha_LR,blockLR);
        GDEBUG_STREAM("prox_LLR end ");
    }
    catch (...)
    {
        GERROR_STREAM("Something broke");
        ho_out.create(out->get_number_of_elements());
        ho_out.fill(1);
    }
    cudaMemcpy(out->get_data_ptr(), ho_out.get_data_ptr(), out->get_number_of_elements() * sizeof(float_complext), cudaMemcpyHostToDevice);
}


template< unsigned int D> void cuLocallyLowRankOperator<D>::gradient(cuNDArray<float_complext>*in, cuNDArray<float_complext>*out, bool accumulate){
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
        GERROR_STREAM("Gradient operator can't be computed");
        //mult_MH_M(in,tmp,false);
        //*tmp *= this->weight_;
    }
    if (accumulate){
	    *out += *tmp;
	    delete tmp;
    }

}

template class  cuLocallyLowRankOperator<1>;
template class  cuLocallyLowRankOperator<2>;
template class  cuLocallyLowRankOperator<3>;
template class  cuLocallyLowRankOperator<4>;
template class  cuLocallyLowRankOperator<5>;