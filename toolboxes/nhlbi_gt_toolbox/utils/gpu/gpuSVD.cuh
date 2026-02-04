//#pragma once

#include <cuNDArray.h>
#include <cuNDArray_math.h>
#include <cuNDArray_elemwise.h>
#include <cuSenseOperator.h>
#include <cuNFFT.h>
#include <hoNDArray.h>

//#include "hoNDArray_iterators.h"
//#include "vector_td_utilities.h"
#include "hoNDArray_elemwise.h"
#include "hoNDArray_reductions.h"

namespace Gadgetron{
    class gpuSVD
    {
    public:
        gpuSVD() = default;
        std::tuple<cuNDArray<float>,cuNDArray<float>,cuNDArray<float>> cuda_DNSgesvd(cuNDArray<float>* A_in, int vectype);
        std::tuple<cuNDArray<float_complext>,cuNDArray<float>,cuNDArray<float_complext>> cuda_DNCgesvd(cuNDArray<float_complext>* A_in, int vectype);
        std::tuple<cuNDArray<float>,cuNDArray<float>,cuNDArray<float>> cuda_DNSgesvdj(cuNDArray<float>* A_in, int vectype);
        std::tuple<cuNDArray<float_complext>,cuNDArray<float>,cuNDArray<float_complext>> cuda_DNCgesvdj(cuNDArray<float_complext>* A_in, int vectype);
        std::tuple<hoNDArray<float>,hoNDArray<float>,hoNDArray<float>> cpu_lapacke_Ssvd(hoNDArray<float>* A_in, int vectype);
        std::tuple<hoNDArray<float_complext>,hoNDArray<float>,hoNDArray<float_complext>> cpu_lapacke_Csvd(hoNDArray<float_complext>* A_in, int vectype);
        void soft_thresh(cuNDArray<float>* x,float thresh);
        cuNDArray<float_complext> apply_SVD(cuNDArray<float_complext> U ,cuNDArray<float> S,cuNDArray<float_complext> Vh);
        cuNDArray<float> apply_SVD(cuNDArray<float> U ,cuNDArray<float> S,cuNDArray<float> Vh);
        cuNDArray<float_complext> batch_LR(cuNDArray<float_complext>* A_in, int vectype,float thresh);
        //void matrixMultiplyCUDA(cuNDArray<float>* X, cuNDArray<float>* Y, cuNDArray<float>* R);
        hoNDArray<float_complext> svd_pixelwise_lapack(hoNDArray<float_complext>* A_in, int vectype);


    };
}

