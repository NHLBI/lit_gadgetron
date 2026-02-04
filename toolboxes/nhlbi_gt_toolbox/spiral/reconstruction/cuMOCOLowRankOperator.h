/** \file cuMOCOLowRankOperator.h
    \brief Low Rank Operator, GPU based.
*/
#pragma once

#include "noncartesian_reconstruction.h"
#include <linearOperator.h>
#include <cuNDArray_operators.h>
#include <cuNDArray.h>
#include <cuNDArray_utils.h>
#include "reconParams.h"
namespace Gadgetron {

template<unsigned int D> class cuMOCOLowRankOperator : public linearOperator<cuNDArray<float_complext>>{

public:
	cuMOCOLowRankOperator() : linearOperator<cuNDArray<float_complext> >(){};

	virtual ~cuMOCOLowRankOperator(){};
	virtual void mult_M(cuNDArray<float_complext>*in,cuNDArray<float_complext>*out,bool accumulate = false );
	virtual void mult_MH(cuNDArray<float_complext>*in,cuNDArray<float_complext>*out,bool accumulate = false );
	virtual void mult_MH_M(cuNDArray<float_complext>*in ,cuNDArray<float_complext>*out,bool accumulate =false);
    virtual void gradient(cuNDArray<float_complext>*in, cuNDArray<float_complext>*out, bool accumulate=false);
    virtual void prox(cuNDArray<float_complext>*in, cuNDArray<float_complext>*out, bool accumulate=false);
	/**
     * Sets the alphaLR of the operator
     * @param[in] alphaLR
     */
    virtual void set_alphaLR( float alphaLR ){ alphaLR_ = alphaLR; }
    virtual void set_prox( bool prox ){ prox_ = prox; }
    virtual void set_refPhase( float refPhase ){ refPhase_ = refPhase; }
    virtual void set_maxIteRegistration( int max_ite_init, int max_ite_r){ max_ite_registration = max_ite_r; max_ite=max_ite_init; use_saved_registration_=true; }
    virtual void set_gpus(std::vector<int> gpus_input){this->gpus = gpus_input;}
    /**
     *
     * @return alphaLR of the operator
     */
    virtual float get_alphaLR(){ return alphaLR_; }
    virtual float get_prox(){ return prox_; }
    virtual float get_refPhase(){ return refPhase_; }
    

    std::tuple<std::vector<cuNDArray<float>>, std::vector<cuNDArray<float>>> register_images_gpu(cuNDArray<float_complext> images_gpu, float referencePhase);
    std::tuple<std::vector<cuNDArray<float>>, std::vector<cuNDArray<float>>> register_images_gpu_image_batch(cuNDArray<float_complext> images_gpu, float referencePhase);
 
    hoNDArray<std::complex<float>> applyDeformations(hoNDArray<std::complex<float>> images_all, std::vector<cuNDArray<float>> deformation);

protected:
	float alphaLR_;
    bool prox_ =true;
    float refPhase_ =0;
    bool use_saved_registration_ = false; // If true, use the saved registration from the class, otherwise register the images again.
    int max_ite_registration = 0; 
    int current_ite = 0; // Current iteration for registration, used to check if we need to register again.
    int max_ite= 0; // Maximum number of iterations.
    std::vector<int> gpus;
    std::vector<cuNDArray<float>> forward_deformation_;
    std::vector<cuNDArray<float>> backward_deformation_;
    bool debug_ = false;

};

}
