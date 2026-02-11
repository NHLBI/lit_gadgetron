#include "NFFTOperator.h"
#include "complext.h"
#include "cuNDArray_blas.h"
#include "cuNDArray_elemwise.h"
#include "cuNDArray_reductions.h"
#include "cuNDArray_utils.h"
#include "cuNFFT.h"

#include "vector_td_utilities.h"
#include <complex>
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

// Std includes
#include "GadgetronTimer.h"
#include "hoArmadillo.h"
#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>
#include <iostream>
#include <noncartesian_reconstruction.h>
#include <reconParams.h>

using namespace Gadgetron;

using testing::Types;

template <typename T> class cuNFFT_test : public ::testing::Test {
  protected:
    virtual void SetUp() {
        // Prep for NUFFT_TEST
        RO = 2500;
        INT = 1100;
        CHA = 8;
        xsize_ = 256;
        ysize_ = 256;
        kernel_width_ = 3;
        oversampling_factor_ = 2.1;

        std::vector<size_t> data_dims = {RO, INT, CHA};
        fake_data = cuNDArray<float_complext>(data_dims);
        data_dims.pop_back();
        fake_dcw = cuNDArray<float>(data_dims);

        hoNDArray<vector_td<float, 2>> fake_traj_ho(data_dims);
        vector_td<float, 2> init_val;
        init_val[0] = 0.1f;
        init_val[1] = 0.1f;
        fake_traj_ho.fill(init_val);

        fake_traj = cuNDArray<vector_td<float, 2>>(fake_traj_ho);
        fill(&fake_dcw, 1.0f);
        fill(&fake_data, complext(1.0f, 0.0f));

        
        
        unsigned int warp_size = cudaDeviceManager::Instance()->warp_size();

        image_dims_.push_back(xsize_);
        image_dims_.push_back(ysize_);

        image_dims_os_ = uint64d2(
            ((static_cast<size_t>(std::ceil(image_dims_[0] * oversampling_factor_)) + warp_size - 1) / warp_size) *
                warp_size,
            ((static_cast<size_t>(std::ceil(image_dims_[1] * oversampling_factor_)) + warp_size - 1) / warp_size) *
                warp_size); // No oversampling is needed in the z-direction for SOS

        recon_dims = {this->image_dims_[0], this->image_dims_[1], CHA};
    }

    cuNDArray<float_complext> fake_data;
    cuNDArray<float> fake_dcw;
    cuNDArray<vector_td<float, 2>> fake_traj;
    size_t RO, INT, Kz, CHA, xsize_, ysize_;
    float kernel_width_, oversampling_factor_;
    uint64d2 image_dims_os_;
    std::vector<size_t> image_dims_;
    boost::shared_ptr<cuNFFT_plan<float, 2>> nfft_plan_;
    std::vector<size_t> recon_dims;
    GadgetronTimer timer_;

};

typedef Types<float_complext> cplxImplementations;

TYPED_TEST_SUITE(cuNFFT_test, cplxImplementations);

TYPED_TEST(cuNFFT_test, cuNFFT_ATOMIC) {
    this->timer_.start("ATOMIC NUFFT");
    std::vector<size_t> flat_dims = {this->fake_traj.get_number_of_elements()};
    cuNDArray<vector_td<float, 2>> flat_traj(flat_dims, this->fake_traj.get_data_ptr());
    this->nfft_plan_ =
        NFFT<cuNDArray, float, 2>::make_plan(from_std_vector<size_t, 2>(this->image_dims_), this->image_dims_os_,
                                             this->kernel_width_, ConvolutionType::ATOMIC);

    {
        this->nfft_plan_->preprocess(flat_traj, NFFT_prep_mode::NC2C);
    }
    auto temp = boost::make_shared<cuNDArray<float_complext>>(this->recon_dims);

    {
        this->nfft_plan_->compute(&this->fake_data, *temp, &this->fake_dcw, NFFT_comp_mode::BACKWARDS_NC2C);
    }

    reconParams recon_params;
        ISMRMRD::MatrixSize ematrixSize;
        ISMRMRD::MatrixSize rmatrixSize;
        ematrixSize = {this->image_dims_[0], this->image_dims_[1], 1};
        rmatrixSize = ematrixSize;
        recon_params.ematrixSize=ematrixSize;
        recon_params.rmatrixSize=rmatrixSize;
        recon_params.oversampling_factor_=2.1;
        recon_params.kernel_width_=3;
        recon_params.RO = this->RO;
        nhlbi_toolbox::reconstruction::noncartesian_reconstruction<2> nr(recon_params);
        GDEBUG_STREAM("Doing DCF estimation nr.estimate_dcf(&fake_traj)");
        auto estdcw = nr.estimate_dcf(&this->fake_traj);
        GDEBUG_STREAM("Doing DCF estimation nr.estimate_dcf(&fake_traj, &estdcw)");
        nr.estimate_dcf(&this->fake_traj, &estdcw);

        EXPECT_TRUE(this->timer_.stop()<2e6);// Test should take less than 2 sec

}

TYPED_TEST(cuNFFT_test, cuNFFT_SPARSE_MATRIX) {
    this->timer_.start("SPARSE_MATRIX NUFFT");
    std::vector<size_t> flat_dims = {this->fake_traj.get_number_of_elements()};
    cuNDArray<vector_td<float, 2>> flat_traj(flat_dims, this->fake_traj.get_data_ptr());
    this->nfft_plan_ =
        NFFT<cuNDArray, float, 2>::make_plan(from_std_vector<size_t, 2>(this->image_dims_), this->image_dims_os_,
                                             this->kernel_width_, ConvolutionType::SPARSE_MATRIX);

    {
        this->nfft_plan_->preprocess(flat_traj, NFFT_prep_mode::NC2C);
    }
    auto temp = boost::make_shared<cuNDArray<float_complext>>(this->recon_dims);

    {
        this->nfft_plan_->compute(&this->fake_data, *temp, &this->fake_dcw, NFFT_comp_mode::BACKWARDS_NC2C);
    }

    reconParams recon_params;
        ISMRMRD::MatrixSize ematrixSize;
        ISMRMRD::MatrixSize rmatrixSize;
        ematrixSize = {this->image_dims_[0], this->image_dims_[1], 1};
        rmatrixSize = ematrixSize;
        recon_params.ematrixSize=ematrixSize;
        recon_params.rmatrixSize=rmatrixSize;
        recon_params.oversampling_factor_=2.1;
        recon_params.kernel_width_=3;
        recon_params.RO = this->RO;
        nhlbi_toolbox::reconstruction::noncartesian_reconstruction<2> nr(recon_params);
        GDEBUG_STREAM("Doing DCF estimation nr.estimate_dcf(&fake_traj)");
        auto estdcw = nr.estimate_dcf(&this->fake_traj);
        GDEBUG_STREAM("Doing DCF estimation nr.estimate_dcf(&fake_traj, &estdcw)");
        nr.estimate_dcf(&this->fake_traj, &estdcw);

        EXPECT_TRUE(this->timer_.stop()<2e6);// Test should take less than 2 sec
}

TYPED_TEST(cuNFFT_test, cuNFFT_STANDARD) {
    this->timer_.start("STANDARD NUFFT");
    std::vector<size_t> flat_dims = {this->fake_traj.get_number_of_elements()};
    cuNDArray<vector_td<float, 2>> flat_traj(flat_dims, this->fake_traj.get_data_ptr());
    this->nfft_plan_ =
        NFFT<cuNDArray, float, 2>::make_plan(from_std_vector<size_t, 2>(this->image_dims_), this->image_dims_os_,
                                             this->kernel_width_, ConvolutionType::STANDARD);
    {
        this->nfft_plan_->preprocess(flat_traj, NFFT_prep_mode::NC2C);
    }

    auto temp = boost::make_shared<cuNDArray<float_complext>>(this->recon_dims);
    {
        this->nfft_plan_->compute(&this->fake_data, *temp, &this->fake_dcw, NFFT_comp_mode::BACKWARDS_NC2C);
    }
    EXPECT_TRUE(this->timer_.stop()<20e6); // Test should take less than 20 sec
}

// 3D NUFFT test case (ATOMIC) to cover the 3D GPU code path.
template <typename T> class cuNFFT_3D_test : public ::testing::Test {
  protected:
    void SetUp() override {
        RO = 128;
        INT = 64;
        Kz = 32;
        CHA = 2;

        xsize_ = 64;
        ysize_ = 64;
        zsize_ = 32;

        kernel_width_ = 3;
        oversampling_factor_ = 2.1;

        std::vector<size_t> data_dims = {RO, INT, Kz, CHA};
        fake_data = cuNDArray<float_complext>(data_dims);
        data_dims.pop_back();
        fake_dcw = cuNDArray<float>(data_dims);

        hoNDArray<vector_td<float, 3>> fake_traj_ho(data_dims);
        vector_td<float, 3> init_val;
        init_val[0] = 0.1f;
        init_val[1] = 0.1f;
        init_val[2] = 0.1f;
        fake_traj_ho.fill(init_val);

        fake_traj = cuNDArray<vector_td<float, 3>>(fake_traj_ho);
        fill(&fake_dcw, 1.0f);
        fill(&fake_data, complext(1.0f, 0.0f));

        const unsigned int warp_size = cudaDeviceManager::Instance()->warp_size();

        image_dims_.clear();
        image_dims_.push_back(xsize_);
        image_dims_.push_back(ysize_);
        image_dims_.push_back(zsize_);

        // Oversample in x/y; keep z un-oversampled to keep memory/runtime modest.
        const size_t x_os =
            ((static_cast<size_t>(std::ceil(image_dims_[0] * oversampling_factor_)) + warp_size - 1) / warp_size) *
            warp_size;
        const size_t y_os =
            ((static_cast<size_t>(std::ceil(image_dims_[1] * oversampling_factor_)) + warp_size - 1) / warp_size) *
            warp_size;
        const size_t z_os = image_dims_[2];

        image_dims_os_ = uint64d3(x_os, y_os, z_os);

        recon_dims = {image_dims_[0], image_dims_[1], image_dims_[2], CHA};
    }

    cuNDArray<float_complext> fake_data;
    cuNDArray<float> fake_dcw;
    cuNDArray<vector_td<float, 3>> fake_traj;
    size_t RO{}, INT{}, Kz{}, CHA{}, xsize_{}, ysize_{}, zsize_{};
    float kernel_width_{}, oversampling_factor_{};
    uint64d3 image_dims_os_;
    std::vector<size_t> image_dims_;
    boost::shared_ptr<cuNFFT_plan<float, 3>> nfft_plan_;
    std::vector<size_t> recon_dims;
    GadgetronTimer timer_;
};

TYPED_TEST_SUITE(cuNFFT_3D_test, cplxImplementations);

TYPED_TEST(cuNFFT_3D_test, cuNFFT_3D_ATOMIC) {
    this->timer_.start("ATOMIC NUFFT 3D");

    std::vector<size_t> flat_dims = {this->fake_traj.get_number_of_elements()};
    cuNDArray<vector_td<float, 3>> flat_traj(flat_dims, this->fake_traj.get_data_ptr());

    this->nfft_plan_ = NFFT<cuNDArray, float, 3>::make_plan(from_std_vector<size_t, 3>(this->image_dims_),
                                                            this->image_dims_os_, this->kernel_width_,
                                                            ConvolutionType::ATOMIC);
    this->nfft_plan_->preprocess(flat_traj, NFFT_prep_mode::NC2C);

    auto temp = boost::make_shared<cuNDArray<float_complext>>(this->recon_dims);
    this->nfft_plan_->compute(&this->fake_data, *temp, &this->fake_dcw, NFFT_comp_mode::BACKWARDS_NC2C);

    const float out_norm = nrm2(temp.get());
    EXPECT_TRUE(std::isfinite(out_norm));
    EXPECT_GT(out_norm, 0.0f);

    EXPECT_TRUE(this->timer_.stop() < 4e6); // <4 sec
}
