#include <cuNDArray.h>
#include <cuNDArray_math.h>
#include <cuNDArray_elemwise.h>
#include <cuSenseOperator.h>
#include <cuNFFT.h>
#include <cuNDArray_math.h>
#include <cuNDArray_elemwise.h>
#include <hoNDArray.h>
#include <hoNDArray_elemwise.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <string>

#include <cuda_runtime.h>

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuComplex.h>
#include <gtest/gtest.h>
#include "gpuSVD.cuh"
#include "cpuSVD.h"
#include "util_functions.h"
#include <boost/random.hpp>

using namespace Gadgetron;
using testing::Types;

class gpu_SVD_test : public ::testing::Test {
protected:
  virtual void SetUp() {
    size_t m=32;
    size_t n=10;
    boost::random::mt19937 rng;
		boost::random::uniform_real_distribution<float> uni(0,1);
    std::vector<size_t> dims_A={m,n};
    ho_A.create(dims_A);
    ho_Ac.create(dims_A);
    
    for (size_t i = 0; i < ho_A.get_number_of_elements(); i++){
      ho_A[i] = uni(rng);
      ho_Ac[i] =float_complext(uni(rng),uni(rng));
    }
    A = cuNDArray<float>(ho_A);
    Ac = cuNDArray<float_complext>(ho_Ac);
    //fill(&A,1.0f);

    size_t x=160;
    size_t y=160;
    size_t coils=34;
    std::vector<size_t> dims_csm={x,y,coils};
    csm.create(dims_csm);
    
    for (size_t i = 0; i < csm.get_number_of_elements(); i++){
      csm[i] =float_complext(uni(rng),uni(rng));
    }

  }
  cuNDArray<float> A;
  cuNDArray<float_complext> Ac;
  hoNDArray<float > ho_A;
  hoNDArray<float_complext > ho_Ac;
  gpuSVD svd;
  hoNDArray<float_complext> csm;
  cpuSVD svd2;
};



TEST_F(gpu_SVD_test,DNSgesvd) {
// Test SVD on GPU
    size_t m = this->A.get_size(0);
    size_t n = this->A.get_size(1);
    size_t k = std::min(m,n);
    float a_1= this->A.at(1);
    
    cuNDArray<float> B(*(this->A.get_dimensions()));

    cudaMemcpy(B.get_data_ptr(),A.get_data_ptr(),this->A.get_number_of_elements()*sizeof(float), cudaMemcpyDefault);
    auto [U,S,Vh]=svd.cuda_DNSgesvd(&(B),1);
    EXPECT_EQ(k,S.get_size(0));
    EXPECT_EQ(m,U.get_size(0));
    EXPECT_EQ(k,U.get_size(1));
    EXPECT_EQ(k,Vh.get_size(0));
    EXPECT_EQ(n,Vh.get_size(1));
    EXPECT_FLOAT_EQ(a_1, this->A.at(1));
    EXPECT_FLOAT_EQ(a_1, B.at(1));
    EXPECT_TRUE(S.at(0) > S.at(1));}
  
  TEST_F(gpu_SVD_test,DNCgesvd){
// Test SVD on GPU
    size_t m = this->Ac.get_size(0);
    size_t n = this->Ac.get_size(1);
    size_t k = std::min(m,n);
    float_complext a_1= this->Ac.at(1);
    
    cuNDArray<float_complext> B(*(this->A.get_dimensions()));

    cudaMemcpy(B.get_data_ptr(),Ac.get_data_ptr(),this->Ac.get_number_of_elements()*sizeof(float_complext), cudaMemcpyDefault);
    auto [U,S,Vh]=svd.cuda_DNCgesvd(&(B),1);
    EXPECT_EQ(k,S.get_size(0));
    EXPECT_EQ(m,U.get_size(0));
    EXPECT_EQ(k,U.get_size(1));
    EXPECT_EQ(k,Vh.get_size(0));
    EXPECT_EQ(n,Vh.get_size(1));
    EXPECT_FLOAT_EQ(real(a_1), real(this->Ac.at(1)));
    EXPECT_FLOAT_EQ(real(a_1), real(B.at(1)));
    EXPECT_TRUE(S.at(0) > S.at(1));}

TEST_F(gpu_SVD_test,DNSgesvdj) {
// Test SVD on GPU
    size_t m = A.get_size(0);
    size_t n = this->A.get_size(1);
    size_t k = std::min(m,n);
    float a_1= A.at(1);
    
    auto [U,S,Vh]=svd.cuda_DNSgesvdj(&(A),1);
    EXPECT_EQ(k,S.get_size(0));
    EXPECT_EQ(m,U.get_size(0));
    EXPECT_EQ(k,U.get_size(1));
    EXPECT_EQ(k,Vh.get_size(0));
    EXPECT_EQ(n,Vh.get_size(1));
    EXPECT_FLOAT_EQ(a_1, this->A.at(1));
    EXPECT_TRUE(S.at(0) > S.at(1));}
  
TEST_F(gpu_SVD_test,DNCgesvdj){
// Test SVD on GPU
    size_t m = this->Ac.get_size(0);
    size_t n = this->Ac.get_size(1);
    size_t k = std::min(m,n);
    float_complext a_1= this->Ac.at(1);
    
    cuNDArray<float_complext> B(*(this->A.get_dimensions()));

    cudaMemcpy(B.get_data_ptr(),Ac.get_data_ptr(),this->Ac.get_number_of_elements()*sizeof(float_complext), cudaMemcpyDefault);
    auto [U,S,Vh]=svd.cuda_DNCgesvdj(&(B),1);
    EXPECT_EQ(k,S.get_size(0));
    EXPECT_EQ(m,U.get_size(0));
    EXPECT_EQ(k,U.get_size(1));
    EXPECT_EQ(k,Vh.get_size(0));
    EXPECT_EQ(n,Vh.get_size(1));
    EXPECT_FLOAT_EQ(real(a_1), real(this->Ac.at(1)));
    EXPECT_FLOAT_EQ(real(a_1), real(B.at(1)));
    EXPECT_TRUE(S.at(0) > S.at(1));}

TEST_F(gpu_SVD_test,lapacke_Csvd) {
// Test SVD on GPU
    size_t m = this->ho_Ac.get_size(0);
    size_t n = this->ho_Ac.get_size(1);
    size_t k = std::min(m,n);
    float_complext a_1= this->ho_Ac.at(1);
    
    hoNDArray<float_complext> B(*(this->ho_Ac.get_dimensions()));

    memcpy(B.get_data_ptr(),ho_Ac.get_data_ptr(),ho_Ac.get_number_of_elements()*sizeof(float_complext));
    auto [U,S,Vh]=svd.cpu_lapacke_Csvd(&(B),1);
    EXPECT_EQ(k,S.get_size(0));
    EXPECT_EQ(m,U.get_size(0));
    EXPECT_EQ(k,U.get_size(1));
    EXPECT_EQ(k,Vh.get_size(0));
    EXPECT_EQ(n,Vh.get_size(1));
    EXPECT_FLOAT_EQ(real(a_1), real(this->Ac.at(1)));
    EXPECT_FLOAT_EQ(real(a_1), real(B.at(1)));
    EXPECT_TRUE(S.at(0) > S.at(1));
    }

TEST_F(gpu_SVD_test,lapacke_Ssvd) {
    // Test SVD on GPU
    size_t m = this->ho_A.get_size(0);
    size_t n = this->ho_A.get_size(1);
    size_t k = std::min(m,n);
    float a_1= this->ho_A.at(1);
    
    hoNDArray<float> B(*(this->ho_A.get_dimensions()));

    memcpy(B.get_data_ptr(),ho_A.get_data_ptr(),ho_A.get_number_of_elements()*sizeof(float));
    auto [U,S,Vh]=svd.cpu_lapacke_Ssvd(&(B),1);
    EXPECT_EQ(k,S.get_size(0));
    EXPECT_EQ(m,U.get_size(0));
    EXPECT_EQ(k,U.get_size(1));
    EXPECT_EQ(k,Vh.get_size(0));
    EXPECT_EQ(n,Vh.get_size(1));
    EXPECT_FLOAT_EQ(a_1, this->ho_A.at(1));
    EXPECT_FLOAT_EQ(a_1, B.at(1));
    EXPECT_TRUE(S.at(0) > S.at(1));
    }

  TEST_F(gpu_SVD_test,soft_thresh) {
    hoNDArray<float> S_test(3);
    float vec[]={5.0,2.5,-6.0};
    for (size_t i = 0; i < S_test.get_number_of_elements(); i++){
      S_test[i] = vec[i];
    }
    cuNDArray<float> S(S_test);
    
    float thresh = 3.0f;
    svd.soft_thresh(&(S),thresh);
    EXPECT_EQ(2.0,S.at(0));
    EXPECT_EQ(0,S.at(1));
    EXPECT_EQ(-3,S.at(2));

    }

TEST_F(gpu_SVD_test,apply_SVD){
// Test apply_SVD on GPU
    // This is a test for the apply_SVD function, which applies the SVD decomposition to reconstruct the original matrix.
    // The input matrix A is decomposed into U, S, and Vh, and then reconstructed as C.
    // The test checks if the first two elements of A and C are equal.
    // This is a simple test to verify that the SVD decomposition and reconstruction works correctly.

    // Create a simple matrix A
    std::vector<float> A_init = {3, 1,1, 3};
    std::vector<float> U_init = {-float(std::sqrt(2.0)) / 2, -float(std::sqrt(2)) / 2,-float(std::sqrt(2)) / 2, float(std::sqrt(2)) / 2};
    std::vector<float> S_init = {4, 2};
    std::vector<float> Vh_init = {-float(std::sqrt(2.0)) / 2, -float(std::sqrt(2)) / 2,-float(std::sqrt(2)) / 2, float(std::sqrt(2)) / 2};;
    cuNDArray<float> A_test({2, 2}, A_init.data());
    
    auto [U,S,Vh]=svd.cuda_DNSgesvd(&(A_test),1);
    for (size_t i = 0; i < U.get_number_of_elements(); i++) {
        EXPECT_FLOAT_EQ(U.at(i), U_init[i]);
    }
    for (size_t i = 0; i < S.get_number_of_elements(); i++) {
        EXPECT_FLOAT_EQ(S.at(i), S_init[i]);
    }
    for (size_t i = 0; i < Vh.get_number_of_elements(); i++) {
        EXPECT_FLOAT_EQ(Vh.at(i), Vh_init[i]);
    }

    cuNDArray<float> C = svd.apply_SVD(U, S, Vh);
    for (size_t i = 0; i < C.get_number_of_elements(); i++) {
        EXPECT_FLOAT_EQ(C.at(i), A_init[i]);
    }
}

TEST_F(gpu_SVD_test,apply_SVD_random){
// Test apply_SVD on GPU
    // This is a test for the apply_SVD function, which applies the SVD decomposition to reconstruct the original matrix.
    // The input matrix A is decomposed into U, S, and Vh, and then reconstructed as C.
    // The test checks if the first two elements of A and C are equal.
    // This is a simple test to verify that the SVD decomposition and reconstruction works correctly.

    // Create a simple matrix A
    
    
    //std::vector<float> A_init ={1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
    //cuNDArray<float> A_test({3, 4}, A_init.data());
    
    auto [U,S,Vh]=svd.cuda_DNSgesvd(&(A),1);
    cuNDArray<float> C = svd.apply_SVD(U, S, Vh);
    for (size_t i = 0; i < C.get_number_of_elements(); i++) {
        EXPECT_NEAR(C.at(i), A[i],1e-3);
    }
    /*
    auto [U,S,Vh]=svd.cuda_DNSgesvdj(&(A_test),1);
    cuNDArray<float> C = svd.apply_SVD(U, S, Vh);
    for (size_t i = 0; i < C.get_number_of_elements(); i++) {
        EXPECT_NEAR(C.at(i), A_init[i],1e-3);
    }
    */
    auto [Ut,St,Vht]=svd.cuda_DNSgesvdj(&(A),1);
    cuNDArray<float> Ct = svd.apply_SVD(Ut, St, Vht);
    for (size_t i = 0; i < Ct.get_number_of_elements(); i++) {
        EXPECT_NEAR(Ct.at(i), A.at(i),1e-3);
    }
    /*
    std::vector<float> A_init_2 ={1, 2, 3, 4, 0, 0, 1, 0, 0, 0, 1, 0};
    cuNDArray<float> A_test2({3, 4}, A_init_2.data());
    auto [U2,S2,Vh2]=svd.cuda_DNSgesvdj(&(A_test2),1);
    cuNDArray<float> C2 = svd.apply_SVD(U2, S2, Vh2);
    for (size_t i = 0; i < C2.get_number_of_elements(); i++) {
        EXPECT_NEAR(C2.at(i), A_init_2[i],1e-3);
    }
    */
}

TEST_F(gpu_SVD_test,apply_SVD_random_cmplx){
// Test apply_SVD on GPU
    auto [U,S,Vh]=svd.cuda_DNCgesvd(&(Ac),1);
    auto C = svd.apply_SVD(U, S, Vh);
    for (size_t i = 0; i < 2; i++) {
        EXPECT_NEAR(real(C.at(i)), real(Ac.at(i)),1e-3);
        EXPECT_NEAR(imag(C.at(i)), imag(Ac.at(i)),1e-3);
    }
  
}

TEST_F(gpu_SVD_test,apply_SVD_random_cmplx_j){
    auto [U,S,Vh]=svd.cuda_DNCgesvdj(&(Ac),1);
    auto C = svd.apply_SVD(U, S, Vh);
    for (size_t i = 0; i < C.get_number_of_elements(); i++) {
        EXPECT_NEAR(real(C.at(i)), real(Ac.at(i)),1e-3);
        EXPECT_NEAR(imag(C.at(i)), imag(Ac.at(i)),1e-3);
    }
}
/*
TEST_F(gpu_SVD_test,apply_SVD_random_cmplx_LR){
    auto C=svd.batch_LR(&(Ac),1,0.0f);
    for (size_t i = 14; i < 20; i++) {
        GDEBUG_STREAM(C.at(i)<< " " <<Ac.at(i))
        EXPECT_NEAR(real(C.at(i)), real(Ac.at(i)),1e-3);
        EXPECT_NEAR(imag(C.at(i)), imag(Ac.at(i)),1e-3);
    }

    size_t m=320*320*320;
    size_t n=10;
    boost::random::mt19937 rng;
	boost::random::uniform_real_distribution<float> uni(0,1);
    std::vector<size_t> dims_A={m,n};
    hoNDArray<float_complext> ho_Ac_bis(dims_A);
    
    for (size_t i = 0; i < ho_Ac_bis.get_number_of_elements(); i++){
      ho_Ac_bis[i] =float_complext(uni(rng),uni(rng));
    }
    auto Ac_bis = cuNDArray<float_complext>(ho_Ac_bis);

    auto C_bis=svd.batch_LR(&(Ac_bis),1,0.0f);
    for (size_t i = 14; i < 20; i++) {
        GDEBUG_STREAM(C_bis.at(i)<< " " <<Ac.at(i))
        EXPECT_NEAR(real(C_bis.at(i)), real(Ac_bis.at(i)),1e-3);
        EXPECT_NEAR(imag(C_bis.at(i)), imag(Ac_bis.at(i)),1e-3);
    }

}
*/

TEST_F(gpu_SVD_test,svd_pixelwise_lapack){

    
    //auto cu_csm = cuNDArray<float_complext>(csm);

    auto Vh=svd2.svd_pixelwise_lapack(&(csm),2);
    std::string out_file = std::string("/opt/data/daudepv/tmp");
    nhlbi_toolbox::utils::write_cpu_nd_array<float_complext>(csm, out_file + std::string("_csm.complex"));
    nhlbi_toolbox::utils::write_cpu_nd_array<float_complext>(Vh, out_file + std::string("_Vh_pixel.complex"));
}