//#pragma once


#include <hoNDArray.h>
#include "hoNDArray_iterators.h"
#include "vector_td_utilities.h"
#include "hoNDArray_elemwise.h"
#include "hoNDArray_utils.h"
#include "hoNDArray_reductions.h"

namespace Gadgetron{
    class cpuSVD
    {
    public:
        cpuSVD() = default;
        std::tuple<hoNDArray<float>,hoNDArray<float>,hoNDArray<float>> cpu_lapacke_Ssvd(hoNDArray<float>* A_in, int vectype);
        std::tuple<hoNDArray<float_complext>,hoNDArray<float>,hoNDArray<float_complext>> cpu_lapacke_Csvd(hoNDArray<float_complext>* A_in, int vectype);
        hoNDArray<float_complext> svd_pixelwise_lapack(hoNDArray<float_complext>* A_in, int vectype);
    };
}

