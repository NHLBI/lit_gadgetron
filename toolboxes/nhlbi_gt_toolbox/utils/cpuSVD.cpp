/*
List of function to computes the singular value decomposition (SVD)
[U, S, VT] = svd(A)
Remark : gesvd only supports m>=n.
-------------------------------------------------------------- 
The SVD is written A = U ∗ S ∗ V H  
where  A (mxn matrix), S is min(m,n)x 1 diagonal matrix, which elements of S are the singular values of A.They are real and non-negative, and are returned in descending order.
U is an m × m unitary matrix, and V is an n × n unitary matrix. 
The first min(m,n) columns of U and V are the left and right singular vectors of A.
-------------------------------------------------------------- 
Documentation extracted from cuSolver API:
gesvdj has the same functionality as gesvd. The difference is that gesvd uses QR algorithm and gesvdj uses Jacobi method. 
The parallelism of Jacobi method gives GPU better performance on small and medium size matrices. 
Moreover the user can configure gesvdj to perform approximation up to certain accuracy.
*/

#include "cpuSVD.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <string>


using namespace Gadgetron;

#ifdef USE_MKL
  #include <mkl.h>
  #include <mkl_lapacke.h>
#endif
#ifdef USE_EIGEN
  #include <Eigen/Dense>
#endif

static double time_now_ms() {
  using clock = std::chrono::high_resolution_clock;
  return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

static char vec_char(int v, bool for_u) {
  (void)for_u; // unused for now
  switch(v) {
    case 0: return 'N'; //Vectors::None
    case 1: return 'S'; //Vectors::Thin
    case 2: default: return 'A'; //Vectors::Full
  }
}

std::tuple<hoNDArray<float>,hoNDArray<float>,hoNDArray<float>> cpuSVD::cpu_lapacke_Ssvd(hoNDArray<float>* A_in, int vectype) {
  int m = A_in->get_size(0);
  int n = A_in->get_size(1);
  hoNDArray<float> A(A_in->get_dimensions());
  memcpy(A.get_data_ptr(), A_in->get_data_ptr(), A_in->get_number_of_bytes());
  int lda = m;
  int k = std::min(m,n);
  size_t Ucols = size_t(vectype==1 ? k : m);
  size_t VTrows = size_t(vectype==1 ? k : n);
  int ldu = m;
  int ldv = VTrows;
  std::vector<size_t> dims_A={size_t(m),size_t(n)};
  std::vector<size_t> dims_S={size_t(k)};
  std::vector<size_t> dims_U={size_t(m),Ucols};
  std::vector<size_t> dims_VT={VTrows,size_t(n)};
  hoNDArray<float> d_S(dims_S);
  hoNDArray<float> d_U(dims_U);
  hoNDArray<float> d_VT(dims_VT);
  hoNDArray<float>d_A(dims_A);
  int info=0;
  #if defined(USE_MKL)
    
    char jobz = vec_char(vectype, true);
    double t0 = time_now_ms();
    info = LAPACKE_sgesdd(LAPACK_COL_MAJOR, jobz, m, n, A.data(), lda, d_S.data(),d_U.data(), ldu,d_VT.data(), ldv);
    double t1 = time_now_ms();  

    if (info != 0) {
      GERROR_STREAM("MKL sgesdd failed, info=" << info);
    } 
    //else {
    // GDEBUG_STREAM("CPU MKL sgesdd (jobz=" << jobz << ") time: " << (t1 - t0) << " ms" );
    //}
    return std::make_tuple(std::move(d_U), std::move(d_S), std::move(d_VT));
  #elif defined(USE_EIGEN)
    /*Code need to be reviewed and tested , missing U,V,S ...*/
    GERROR_STREAM("CPU SVD with Eigen not implemented yet");
    /*
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> M(A_in.data(), m, n);
    auto t0 = std::chrono::high_resolution_clock::now();
    unsigned flags = 0;
    if (vec == Vectors::None) flags = 0; else if (vec == Vectors::Thin) flags = Eigen::ComputeThinU | Eigen::ComputeThinV; else flags = Eigen::ComputeFullU | Eigen::ComputeFullV;
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(M, flags);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "CPU Eigen JacobiSVD (vec) time: " << ms << " ms" << std::endl;
    return ms;
    */
  #else
    GERROR_STREAM("CPU SVD disabled (no MKL/Eigen)");
  #endif
  }

  std::tuple<hoNDArray<float_complext>,hoNDArray<float>,hoNDArray<float_complext>> cpuSVD::cpu_lapacke_Csvd(hoNDArray<float_complext>* A_in, int vectype) {
  int m = A_in->get_size(0);
  int n = A_in->get_size(1);
  hoNDArray<float_complext> A(A_in->get_dimensions());
  memcpy(A.get_data_ptr(), A_in->get_data_ptr(), A_in->get_number_of_bytes());
  int lda = m;
  int k = std::min(m,n);
  size_t Ucols = size_t(vectype==1 ? k : m);
  size_t VTrows = size_t(vectype==1 ? k : n);
  int ldu = m;
  int ldv = VTrows;
  std::vector<size_t> dims_A={size_t(m),size_t(n)};
  std::vector<size_t> dims_S={size_t(k)};
  std::vector<size_t> dims_U={size_t(m),Ucols};
  std::vector<size_t> dims_VT={VTrows,size_t(n)};
  hoNDArray<float> d_S(dims_S);
  hoNDArray<float_complext> d_U(dims_U);
  hoNDArray<float_complext> d_VT(dims_VT);
  hoNDArray<float_complext>d_A(dims_A);
  int info=0;
  #if defined(USE_MKL)
    
    char jobz = vec_char(vectype, true);
    double t0 = time_now_ms();
    info = LAPACKE_cgesdd(LAPACK_COL_MAJOR, jobz, m, n,reinterpret_cast<MKL_Complex8*>(A.data()), lda, d_S.data(),reinterpret_cast<MKL_Complex8*>(d_U.data()), ldu,reinterpret_cast<MKL_Complex8*>(d_VT.data()), ldv);
    double t1 = time_now_ms();  

    if (info != 0) {
      GERROR_STREAM("MKL sgesdd failed, info=" << info);
    } 
    //else {
    //  GDEBUG_STREAM("CPU MKL sgesdd (jobz=" << jobz << ") time: " << (t1 - t0) << " ms" );
    //}
    return std::make_tuple(std::move(d_U), std::move(d_S), std::move(d_VT));
  #elif defined(USE_EIGEN)
    /*Code need to be reviewed and tested , missing U,V,S ...*/
    GERROR_STREAM("CPU SVD with Eigen not implemented yet");
    /*
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> M(A_in.data(), m, n);
    auto t0 = std::chrono::high_resolution_clock::now();
    unsigned flags = 0;
    if (vec == Vectors::None) flags = 0; else if (vec == Vectors::Thin) flags = Eigen::ComputeThinU | Eigen::ComputeThinV; else flags = Eigen::ComputeFullU | Eigen::ComputeFullV;
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(M, flags);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "CPU Eigen JacobiSVD (vec) time: " << ms << " ms" << std::endl;
    return ms;
    */
  #else
    GERROR_STREAM("CPU SVD disabled (no MKL/Eigen)");
  #endif
  }

hoNDArray<float_complext> cpuSVD::svd_pixelwise_lapack(hoNDArray<float_complext>* A_in, int vectype){
  auto x = A_in->get_size(0);
  auto y = A_in->get_size(1);
  //auto coils = A_in->get_size(2);


  std::vector<size_t> dim_A = *A_in->get_dimensions();
  size_t dim_A_size = A_in->get_number_of_dimensions();
  auto coils = dim_A[dim_A_size-1];
  //auto xyz_pixels= x*y;//
  auto xyz_pixels=std::accumulate(dim_A.begin(),dim_A.end()-1, 1, std::multiplies<uint64_t>());
  hoNDArray<float_complext>  ho_pixelwise({size_t(xyz_pixels),coils});
  memcpy(ho_pixelwise.get_data_ptr(),A_in->get_data_ptr(),A_in->get_number_of_elements()*sizeof(float_complext));
  std::vector<size_t> permute_order = {1,0};
  ho_pixelwise = permute(ho_pixelwise, permute_order); // Transpose to (coils x xyz_pixels) #not working 
  hoNDArray<float_complext> hoV_pixelwise({coils,coils,size_t(xyz_pixels)});
  //GDEBUG_STREAM("MAX thread" << omp_get_max_threads() << " xyz_pixels: " << xyz_pixels << " coils: " << coils);
  #pragma omp parallel for
  for (int i = 0; i < xyz_pixels; i++) {
      hoNDArray<float_complext> ho_onepixel({1,coils},ho_pixelwise.data()+i*coils);
      hoNDArray<float_complext> hoV_onepixel({coils,coils},hoV_pixelwise.data()+i*coils*coils);
      auto [U, S, Vh] = cpu_lapacke_Csvd(&ho_onepixel, vectype);
      //GDEBUG_STREAM("SVD pixelwise lapack: i=" << i << " U size: " << Vh.get_size(0) << "x" << Vh.get_size(1))
      // Copy Vh to hoV_pixelwise
      memcpy(hoV_onepixel.get_data_ptr(),Vh.get_data_ptr(),Vh.get_number_of_elements()*sizeof(float_complext));
      }
  
  return std::move(hoV_pixelwise);
}


  