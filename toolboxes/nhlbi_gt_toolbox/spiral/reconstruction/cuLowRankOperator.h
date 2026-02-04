/** \file cuLowRankOperator.h
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

template<unsigned int D> class cuLowRankOperator : public linearOperator<cuNDArray<float_complext>>{

public:
	cuLowRankOperator() : linearOperator<cuNDArray<float_complext> >(){};

	virtual ~cuLowRankOperator(){};
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
    /**
     *
     * @return alphaLR of the operator
     */
    virtual float get_alphaLR(){ return alphaLR_; }
    virtual float get_prox(){ return prox_; }

protected:
	float alphaLR_;
    bool prox_ =true;
};

}
