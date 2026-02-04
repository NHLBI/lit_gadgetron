#pragma once

#include "gpBbSolver_lit.h"
#include <cuNDArray_operators.h>
#include <cuNDArray_elemwise.h>
#include <cuNDArray_blas.h>
#include <real_utilities.h>
#include <vector_td_utilities.h>


#include <cuSolverUtils.h>
using namespace Gadgetron;
namespace Gadgetron{

  template <class T> class cuGpBbSolver_lit : public Gadgetron::gpBbSolver_lit<cuNDArray<T> >
  {
  public:

    cuGpBbSolver_lit() : gpBbSolver_lit<cuNDArray<T> >() {}
    virtual ~cuGpBbSolver_lit() {}
  };
}
