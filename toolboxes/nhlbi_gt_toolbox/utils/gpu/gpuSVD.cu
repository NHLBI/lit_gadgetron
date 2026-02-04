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

#include "gpuSVD.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <string>

#include <thrust/functional.h>
#include <cuda_runtime.h>
#include "CUBLASContextProvider.h"
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuComplex.h>

using namespace Gadgetron;

#ifdef USE_MKL
  #include <mkl.h>
  #include <mkl_lapacke.h>
#endif
#ifdef USE_EIGEN
  #include <Eigen/Dense>
#endif

#define CUDA_CHECK(call) do { \
  cudaError_t _e = (call); \
  if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
    std::exit(EXIT_FAILURE); \
  } \
} while(0)

#define CUSOLVER_CHECK(call) do { \
  cusolverStatus_t _e = (call); \
  if (_e != CUSOLVER_STATUS_SUCCESS) { \
    fprintf(stderr, "cuSOLVER error %s:%d: %d\n", __FILE__, __LINE__, (int)_e); \
    std::exit(EXIT_FAILURE); \
  } \
} while(0)

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


  struct cuNDA_soft_thresh : public thrust::unary_function<float,float>
  {
  cuNDA_soft_thresh( float _thresh ) : thresh(_thresh) {}
  __device__ float operator()(const float &x) const 
  {
      
      return (abs(x)-thresh <= 0) ? 0.0 : (( x > thresh) ? x-thresh : x+thresh); 
  }
  float thresh;
  };

enum class Mode { BOTH, CPU, GPU };

std::tuple<cuNDArray<float>,cuNDArray<float>,cuNDArray<float>> gpuSVD::cuda_DNSgesvd(cuNDArray<float>* A_in, int vectype){
  /*
  A_in: input matrix (m x n)
  vectype: type of singular vectors to compute
           0: None (no U or VT)
           1: Thin (U is m x k, VT is k x n) with k=min(m,n)
           2: Full (U is m x m, VT is n x n)
  returns: tuple of (U, S, VT)
  */
  cudaSetDevice(A_in->get_device());
  int m = A_in->get_size(0);
  int n = A_in->get_size(1);
  if (m < n) {
    GERROR_STREAM("gesvd only supports m >= n, but got m=" << m << ", n=" << n);
  }
  int lda = m;
  int k = std::min(m,n);
  size_t Ucols = size_t(vectype==1 ? k : m);
  size_t VTrows = size_t(vectype==1 ? k : n);

  std::vector<size_t> dims_A={size_t(m),size_t(n)};
  std::vector<size_t> dims_S={size_t(k)};
  std::vector<size_t> dims_U={size_t(m),Ucols};
  std::vector<size_t> dims_VT={VTrows,size_t(n)};
  cuNDArray<float> d_S(dims_S);
  cuNDArray<float> d_U(dims_U);
  cuNDArray<float> d_VT(dims_VT);
  cuNDArray<float>d_A(dims_A);

  CUDA_CHECK(cudaMemcpy(d_A.get_data_ptr(),A_in->get_data_ptr(),A_in->get_number_of_elements()*sizeof(float), cudaMemcpyDefault));
  cusolverDnHandle_t handle; CUSOLVER_CHECK(cusolverDnCreate(&handle));

  int lwork = 0; int *d_info=nullptr; 
  CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
  CUSOLVER_CHECK(cusolverDnSgesvd_bufferSize(handle, m, n, &lwork));
  float *d_work=nullptr; 
  CUDA_CHECK(cudaMalloc(&d_work, sizeof(float) * lwork));

  cudaEvent_t start, stop; CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  signed char jobu = vec_char(vectype, true);
  signed char jobvt = vec_char(vectype, false);
  CUSOLVER_CHECK(cusolverDnSgesvd(handle, jobu, jobvt, m, n, d_A.data(), lda, d_S.data(),
                                  d_U.data(), (jobu=='S'? m : (jobu=='A'? m : 1)),
                                  d_VT.data(), (jobvt=='S'? k : (jobvt=='A'? n : 1)),
                                  d_work, lwork, nullptr, d_info));
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms=0.f; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  int info_h=0; CUDA_CHECK(cudaMemcpy(&info_h, d_info, sizeof(int), cudaMemcpyDeviceToHost));
  if (info_h != 0) std::cerr << "cuSOLVER Sgesvd info=" << info_h << std::endl;
  std::cout << "GPU cuSOLVER gesvd (float) time: " << ms << " ms" << std::endl;

  CUDA_CHECK(cudaFree(d_work)); CUDA_CHECK(cudaFree(d_info));
  CUSOLVER_CHECK(cusolverDnDestroy(handle));
  d_A.clear();

  return std::make_tuple(std::move(d_U), std::move(d_S), std::move(d_VT));
}

std::tuple<cuNDArray<float_complext>,cuNDArray<float>,cuNDArray<float_complext>> gpuSVD::cuda_DNCgesvd(cuNDArray<float_complext>* A_in, int vectype){
  /*
  A_in: input matrix (m x n)
  vectype: type of singular vectors to compute
           0: None (no U or VT)
           1: Thin (U is m x k, VT is k x n) with k=min(m,n)
           2: Full (U is m x m, VT is n x n)
  returns: tuple of (U, S, VT)
  */
  cudaSetDevice(A_in->get_device());
  int m = A_in->get_size(0);
  int n = A_in->get_size(1);
  if (m < n) {
    GERROR_STREAM("gesvd only supports m >= n, but got m=" << m << ", n=" << n);
  }
  int lda = m;
  int k = std::min(m,n);
  size_t Ucols = size_t(vectype==1 ? k : m);
  size_t VTrows = size_t(vectype==1 ? k : n);

  std::vector<size_t> dims_A={size_t(m),size_t(n)};
  std::vector<size_t> dims_S={size_t(k)};
  std::vector<size_t> dims_U={size_t(m),Ucols};
  std::vector<size_t> dims_VT={VTrows,size_t(n)};
  cuNDArray<float> d_S(dims_S);
  cuNDArray<float_complext> d_U(dims_U);
  cuNDArray<float_complext> d_VT(dims_VT);
  cuNDArray<float_complext>d_A(dims_A);

  CUDA_CHECK(cudaMemcpy(d_A.get_data_ptr(),A_in->get_data_ptr(),A_in->get_number_of_elements()*sizeof(float_complext), cudaMemcpyDefault));
  cusolverDnHandle_t handle; CUSOLVER_CHECK(cusolverDnCreate(&handle));

  int lwork = 0; int *d_info=nullptr; 
  CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
  CUSOLVER_CHECK(cusolverDnCgesvd_bufferSize(handle, m, n, &lwork));
  cuNDArray<float_complext>d_work({size_t(lwork)});

  cudaEvent_t start, stop; CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  signed char jobu = vec_char(vectype, true);
  signed char jobvt = vec_char(vectype, false);
  CUSOLVER_CHECK(cusolverDnCgesvd(handle, jobu, jobvt, m, n, reinterpret_cast<cuComplex*>(d_A.data()), lda,(d_S.data()),
                                  reinterpret_cast<cuComplex*>(d_U.data()), (jobu=='S'? m : (jobu=='A'? m : 1)),
                                  reinterpret_cast<cuComplex*>(d_VT.data()), (jobvt=='S'? k : (jobvt=='A'? n : 1)),
                                  reinterpret_cast<cuComplex*>(d_work.data()), lwork, nullptr, d_info));
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms=0.f; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  int info_h=0; CUDA_CHECK(cudaMemcpy(&info_h, d_info, sizeof(int), cudaMemcpyDeviceToHost));
  if (info_h != 0) std::cerr << "cuSOLVER Sgesvd info=" << info_h << std::endl;
  std::cout << "GPU cuSOLVER gesvd (float) time: " << ms << " ms" << std::endl;
  d_work.clear();
  CUDA_CHECK(cudaFree(d_info));
  CUSOLVER_CHECK(cusolverDnDestroy(handle));
  d_A.clear();
  

  return std::make_tuple(std::move(d_U), std::move(d_S), std::move(d_VT));
}

std::tuple<cuNDArray<float>,cuNDArray<float>,cuNDArray<float>> gpuSVD::cuda_DNSgesvdj(cuNDArray<float>* A_in, int vectype){
  /*
  A_in: input matrix (m x n)
  vectype: type of singular vectors to compute
           0: None (no U or VT)
           1: Thin (U is m x k, VT is k x n) with k=min(m,n)
           2: Full (U is m x m, VT is n x n)
  returns: tuple of (U, S, VT)
  */
  if (vectype == 0){
    GERROR_STREAM("gesvdj (vectors=none) is not implemented ")
  }
  // Default tolerance and max sweeps can be tuned
  const float tol = 1e-5f; const int max_sweeps = 100; const int sort_svd = 1;
  int econ = (vectype==1) ? 1 : 0;

  cudaSetDevice(A_in->get_device());
  int m = A_in->get_size(0);
  int n = A_in->get_size(1);
  int lda = m;
  int ldu = m;
  int ldv = n;
  int k = std::min(m,n);
  size_t Ucols = size_t(vectype==1 ? k : m);
  size_t Vcols = size_t(vectype==1 ? k : n);

  std::vector<size_t> dims_A={size_t(m),size_t(n)};
  std::vector<size_t> dims_S={size_t(k)};
  std::vector<size_t> dims_U={size_t(m),Ucols};
  std::vector<size_t> dims_V={size_t(n),Vcols};
  cuNDArray<float> d_S(dims_S);
  cuNDArray<float> d_U(dims_U);
  cuNDArray<float> d_V(dims_V);
  cuNDArray<float>d_A(dims_A);

  CUDA_CHECK(cudaMemcpy(d_A.get_data_ptr(),A_in->get_data_ptr(),A_in->get_number_of_elements()*sizeof(float), cudaMemcpyDefault));
  cusolverDnHandle_t handle; CUSOLVER_CHECK(cusolverDnCreate(&handle));

  gesvdjInfo_t params; CUSOLVER_CHECK(cusolverDnCreateGesvdjInfo(&params));
  CUSOLVER_CHECK(cusolverDnXgesvdjSetTolerance(params, tol));
  CUSOLVER_CHECK(cusolverDnXgesvdjSetMaxSweeps(params, max_sweeps));
  CUSOLVER_CHECK(cusolverDnXgesvdjSetSortEig(params, sort_svd));

  cusolverEigMode_t mode = CUSOLVER_EIG_MODE_VECTOR;
  
  int lwork = 0; int *d_info=nullptr; CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
  
  CUSOLVER_CHECK(cusolverDnSgesvdj_bufferSize(handle, mode, econ, m, n, d_A.data(), lda, d_S.data(), d_U.data(), ldu, d_V.data(), ldv, &lwork, params));
  cuNDArray<float>d_work({size_t(lwork)});

  cudaEvent_t start, stop; CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  CUSOLVER_CHECK(cusolverDnSgesvdj(handle, mode, econ, m, n, (d_A.data()), lda,(d_S.data()), (d_U.data()), ldu, (d_V.data()), ldv, (d_work.data()), lwork, d_info, params));
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms=0.f; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  int info_h=0; CUDA_CHECK(cudaMemcpy(&info_h, d_info, sizeof(int), cudaMemcpyDeviceToHost));
  if (info_h != 0) std::cerr << "cuSOLVER Sgesvd info=" << info_h << std::endl;
  std::cout << "GPU cuSOLVER gesvd (float) time: " << ms << " ms" << std::endl;
  CUSOLVER_CHECK(cusolverDnDestroyGesvdjInfo(params));
  d_work.clear();
  CUDA_CHECK(cudaFree(d_info));
  CUSOLVER_CHECK(cusolverDnDestroy(handle));
  d_A.clear();
  d_V = permute(d_V, {1, 0}); // Transpose because cuSOLVER gesvdj returns V, we need V^T
  return std::make_tuple(std::move(d_U), std::move(d_S), std::move(d_V));
}
cuNDArray<float_complext> gpuSVD::batch_LR(cuNDArray<float_complext>* A_in, int vectype,float thresh){
  /*
  A_in: input matrix (m x n)
  vectype: type of singular vectors to compute
           0: None (no U or VT)
           1: Thin (U is m x k, VT is k x n) with k=min(m,n)
           2: Full (U is m x m, VT is n x n)
  thresh: threshold for soft-thresholding
  returns: cuNDArray<float_complext> result of SVD application
  */
  auto nele= A_in->get_number_of_elements();
  auto nframes= A_in->get_size(1);
  float svd_max_number= 1e8;
  cudaSetDevice(A_in->get_device()); 
  size_t batch_max =size_t(svd_max_number/float(nframes));
  size_t batch_size = nframes*batch_max;
  size_t N=size_t(std::ceil(float(nele)/float(batch_size)));
  cuNDArray<float_complext> R;

  GDEBUG_STREAM("Batch SVD: N=" << N << " nframes=" << nframes << " nele=" << nele << " svd_max_number=" << svd_max_number);
  if (N==1){ 
    auto [U, S, Vh] = cuda_DNCgesvd(A_in, vectype);
    float SMax = max(&S);
    soft_thresh(&S, thresh*SMax);
    R=apply_SVD(U,S,Vh);
  }else{
    cuNDArray<float_complext> d_A(A_in->get_dimensions());
    CUDA_CHECK(cudaMemcpy(d_A.get_data_ptr(),A_in->get_data_ptr(),A_in->get_number_of_elements()*sizeof(float_complext), cudaMemcpyDefault));
  
    d_A = permute(d_A, {1, 0}); // Transpose to (n x m) for SVD compatibility
    R.create(d_A.get_dimensions()); // Initialize R with zeros
    fill(&R,float_complext(0.0f,0.0f)); // Fill R with zeros
    //R= permute(R, {1, 0}); // Transpose back to (m x n)
    for (size_t i = 0; i < N; ++i) {
      size_t start = i * batch_size;
      size_t end = (start + batch_size)> nele ? nele : (start + batch_size);
      size_t batch_i= end==nele ? size_t(float(end - start) / float(nframes))  : batch_max;
      batch_size = nframes * batch_i; // Update batch_size for the next iteration
    GDEBUG_STREAM("Batch " << i << ": start=" << start << ", end=" << end << ", batch_i=" << batch_i);
    cuNDArray<float_complext> A_batch({nframes,batch_i});
    CUDA_CHECK(cudaMemcpy(A_batch.get_data_ptr(),d_A.get_data_ptr()+start,A_batch.get_number_of_elements()*sizeof(float_complext), cudaMemcpyDefault));
    auto Rview = cuNDArray<float_complext>({nframes,batch_i}, R.data()+start);
    A_batch = permute(A_batch,{1, 0}); // Transpose to (bacth_i x nframes)
    GDEBUG_STREAM("A_batch size: " << A_batch.get_size(0) << " x " << A_batch.get_size(1));
    auto [U, S, Vh] = cuda_DNCgesvd(&A_batch, vectype);
    // Apply soft thresholding to singular values
    float SMax = max(&S);
    GDEBUG_STREAM("SMax: " << SMax << " " <<thresh << " " << S.at(0));  
    soft_thresh(&S, thresh*SMax);
    Rview += permute(apply_SVD(U, S, Vh),{1,0});
    U.clear();
    S.clear();
    Vh.clear();

  }
  // After processing all batches, we need to transpose R back to (m x n)
  R= permute(R, {1, 0}); // Transpose back to (m x n)
  d_A.clear();
  }
  
  return std::move(R);
}
std::tuple<cuNDArray<float_complext>,cuNDArray<float>,cuNDArray<float_complext>> gpuSVD::cuda_DNCgesvdj(cuNDArray<float_complext>* A_in, int vectype){
  /*
  A_in: input matrix (m x n)
  vectype: type of singular vectors to compute
           0: None (no U or VT)
           1: Thin (U is m x k, VT is k x n) with k=min(m,n)
           2: Full (U is m x m, VT is n x n)
  returns: tuple of (U, S, VT)
  */
  if (vectype == 0){
    GERROR_STREAM("gesvdj (vectors=none) is not implemented ")
    }

  // Default tolerance and max sweeps can be tuned
  const float tol = 1e-5f; const int max_sweeps = 100; const int sort_svd = 1;
  int econ = (vectype==1) ? 1 : 0;
  cudaSetDevice(A_in->get_device());
  int m = A_in->get_size(0);
  int n = A_in->get_size(1);
  int lda = m;
  int ldu = m;
  int ldv = n;
  int k = std::min(m,n);
  
  size_t Ucols = size_t(vectype==1 ? k : m);
  size_t Vcols = size_t(vectype==1 ? k : n);

  std::vector<size_t> dims_A={size_t(m),size_t(n)};
  std::vector<size_t> dims_S={size_t(k)};
  std::vector<size_t> dims_U={size_t(m),Ucols};
  std::vector<size_t> dims_V={size_t(n),Vcols};
  cuNDArray<float> d_S(dims_S);
  cuNDArray<float_complext> d_U(dims_U);
  cuNDArray<float_complext> d_V(dims_V);
  cuNDArray<float_complext>d_A(dims_A);

  CUDA_CHECK(cudaMemcpy(d_A.get_data_ptr(),A_in->get_data_ptr(),A_in->get_number_of_elements()*sizeof(float_complext), cudaMemcpyDefault));
  cusolverDnHandle_t handle; CUSOLVER_CHECK(cusolverDnCreate(&handle));

  gesvdjInfo_t params; CUSOLVER_CHECK(cusolverDnCreateGesvdjInfo(&params));
  CUSOLVER_CHECK(cusolverDnXgesvdjSetTolerance(params, tol));
  CUSOLVER_CHECK(cusolverDnXgesvdjSetMaxSweeps(params, max_sweeps));
  CUSOLVER_CHECK(cusolverDnXgesvdjSetSortEig(params, sort_svd));

  
  cusolverEigMode_t mode = CUSOLVER_EIG_MODE_VECTOR;
  
  int lwork = 0; int *d_info=nullptr; CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
  
  CUSOLVER_CHECK(cusolverDnCgesvdj_bufferSize(handle, mode, econ, m, n, reinterpret_cast<const cuComplex*>(d_A.data()), lda,
   (d_S.data()), 
   reinterpret_cast<const cuComplex*>(d_U.data()), ldu, 
   reinterpret_cast<const cuComplex*>(d_V.data()), ldv,
    &lwork, params));
  cuNDArray<float_complext>d_work({size_t(lwork)});

  cudaEvent_t start, stop; CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  CUSOLVER_CHECK(cusolverDnCgesvdj(handle, mode, econ, m, n, reinterpret_cast<cuComplex*>(d_A.data()), lda, 
  (d_S.data()), 
  reinterpret_cast<cuComplex*>(d_U.data()), ldu, 
  reinterpret_cast<cuComplex*>(d_V.data()), ldv, 
  reinterpret_cast<cuComplex*>(d_work.data()), 
  lwork, d_info, params));
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms=0.f; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  int info_h=0; CUDA_CHECK(cudaMemcpy(&info_h, d_info, sizeof(int), cudaMemcpyDeviceToHost));
  if (info_h != 0) std::cerr << "cuSOLVER Sgesvd info=" << info_h << std::endl;
  std::cout << "GPU cuSOLVER gesvd (float) time: " << ms << " ms" << std::endl;
  CUSOLVER_CHECK(cusolverDnDestroyGesvdjInfo(params));
  d_work.clear();
  CUDA_CHECK(cudaFree(d_info));
  CUSOLVER_CHECK(cusolverDnDestroy(handle));
  d_A.clear();
  d_V = permute(d_V, {1, 0}); // Transpose because cuSOLVER gesvdj returns V, we need V^T
  d_V=*conj(&d_V);
  return std::make_tuple(std::move(d_U), std::move(d_S), std::move(d_V));
}

std::tuple<hoNDArray<float>,hoNDArray<float>,hoNDArray<float>> gpuSVD::cpu_lapacke_Ssvd(hoNDArray<float>* A_in, int vectype) {
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

  std::tuple<hoNDArray<float_complext>,hoNDArray<float>,hoNDArray<float_complext>> gpuSVD::cpu_lapacke_Csvd(hoNDArray<float_complext>* A_in, int vectype) {
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

  void gpuSVD::soft_thresh(cuNDArray<float>* x,float thresh){
      /*Soft threshold.
          Performs:
          .. math::(| x | - \lambda)_+  \text{sgn}(x)

      Args:
          lamda (float, or array): Threshold parameter.
          input (array)

      Returns:
          cuNDArray<float> : soft-thresholded result.
      */
  if( x == 0x0 )
      throw std::runtime_error("Gadgetron::clamp_min(): Invalid input array");
  
  thrust::device_ptr<float> xPtr = x->get_device_ptr();
  thrust::transform(xPtr,xPtr+x->get_number_of_elements(),xPtr,cuNDA_soft_thresh(thresh));
  }

cuNDArray<float> gpuSVD::apply_SVD(cuNDArray<float> U ,cuNDArray<float> S,cuNDArray<float> Vh){
    /*
    Args:
        U (cuNDArray<float_complext>): Left singular vectors.
        S (cuNDArray<float>): Singular values.
        Vh (cuNDArray<float_complext>): Right singular vectors (conjugate transpose).
    Returns:
        cuNDArray<float_complext>: Result of the SVD application.
    */
    GDEBUG_STREAM("Device U"<< U.get_device() << " S " << S.get_device() << " Vh " << Vh.get_device());
    cudaSetDevice(U.get_device());
    size_t m = U.get_size(0);
    size_t s = S.get_size(0);
    size_t n = Vh.get_size(1);
    hoNDArray<float> hoS_diag({s,s});
    cuNDArray<float> R({m,n});
    cuNDArray<float> W(Vh.get_dimensions());
    // Check dimensions
    GDEBUG_STREAM("Applying SVD with dimensions: U (" << m << "x" << U.get_size(1) << "), S (" << S.get_size(0) << "), Vh (" << Vh.get_size(0) << "x" << n << ")");

    // Create diagonal matrix from S
    hoS_diag.fill(float(0.0));
    for (size_t i = 0; i < S.get_number_of_elements(); i++) {
        hoS_diag[i * S.get_size(0) + i] = float(S.at(i));
    }

    cuNDArray<float> S_diag(hoS_diag);

    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, int(s),int(n), int(s), 
    (&alpha), 
    (S_diag.get_data_ptr()), int(s),
    (Vh.get_data_ptr()), int(s), 
    (&beta), 
    (W.get_data_ptr()), int(s));
    
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, int(m),int(n), int(n), 
    (&alpha), 
    (U.get_data_ptr()), int(m),
    (W.get_data_ptr()), int(n), 
    (&beta), 
    (R.get_data_ptr()), int(m));

    cublasDestroy(handle);
    W.clear();
    S_diag.clear();
    return std::move(R);

  }
  
cuNDArray<float_complext> gpuSVD::apply_SVD(cuNDArray<float_complext> U ,cuNDArray<float> S,cuNDArray<float_complext> Vh){
  cudaSetDevice(U.get_device());
  size_t m = U.get_size(0);
  size_t s = S.get_size(0);
  size_t n = Vh.get_size(1);
  cuNDArray<float_complext> R({m,n});
  cuNDArray<float_complext> W(Vh.get_dimensions());
  // Create diagonal matrix from S
  hoNDArray<float_complext> hoS_diag({s,s});
  hoS_diag.fill(float_complext(0.0f, 0.0f));
  for (size_t i = 0; i < S.get_number_of_elements(); i++) {
      hoS_diag[i * S.get_size(0) + i] = float_complext(S.at(i),0.0f);
  }
  cuNDArray<float_complext> S_diag(hoS_diag);

  cublasHandle_t handle;
  cublasCreate(&handle);
  const float_complext alpha = float_complext(1.0f, 0.0f);
  const float_complext beta = float_complext(0.0f, 0.0f);
  cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, int(s),int(n), int(s), 
  reinterpret_cast<const cuComplex*>(&alpha), 
  reinterpret_cast<const cuComplex*>(S_diag.get_data_ptr()), int(s),
  reinterpret_cast<const cuComplex*>(Vh.get_data_ptr()), int(s), 
  reinterpret_cast<const cuComplex*>(&beta), 
  reinterpret_cast<cuComplex*>(W.get_data_ptr()), int(s));
  
  cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, int(m),int(n), int(n), 
  reinterpret_cast<const cuComplex*>(&alpha), 
  reinterpret_cast<const cuComplex*>(U.get_data_ptr()), int(m),
  reinterpret_cast<const cuComplex*>(W.get_data_ptr()), int(n), 
  reinterpret_cast<const cuComplex*>(&beta), 
  reinterpret_cast<cuComplex*>(R.get_data_ptr()), int(m));

  cublasDestroy(handle);
  W.clear();
  S_diag.clear();
  return std::move(R);

}




hoNDArray<float_complext> gpuSVD::svd_pixelwise_lapack(hoNDArray<float_complext>* A_in, int vectype){
  auto x = A_in->get_size(0);
  auto y = A_in->get_size(1);
  //auto coils = A_in->get_size(2);

  GDEBUG_STREAM("Not working yet, need to fix lapacke SVD for permute");
  std::vector<size_t> dim_A = *A_in->get_dimensions();
  size_t dim_A_size = A_in->get_number_of_dimensions();
  auto coils = dim_A[dim_A_size-1];
  //auto xyz_pixels= x*y;//
  auto xyz_pixels=std::accumulate(dim_A.begin(),dim_A.end()-1, 1, std::multiplies<uint64_t>());
  hoNDArray<float_complext>  ho_pixelwise({size_t(xyz_pixels),coils});
  memcpy(ho_pixelwise.get_data_ptr(),A_in->get_data_ptr(),A_in->get_number_of_elements()*sizeof(float_complext));
  std::vector<size_t> permute_order = {1,0};
  //ho_pixelwise = permute(ho_pixelwise, permute_order); // Transpose to (coils x xyz_pixels) #not working 
  hoNDArray<float_complext> hoV_pixelwise({coils,coils,size_t(xyz_pixels)});
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


  