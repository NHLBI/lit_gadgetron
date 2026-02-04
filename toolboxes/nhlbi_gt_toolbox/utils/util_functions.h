#pragma once
#include <mri_core_grappa.h>
#include <vector_td_utilities.h>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <cgSolver.h>
#include <hoNDArray_math.h>
#include <hoNDArray_utils.h>
#include <hoNDArray_elemwise.h>
#include <hoNFFT.h>
#include <hoNDFFT.h>
#include <numeric>
#include <random>
#include <NonCartesianTools.h>
#include <NFFTOperator.h>
#include <sstream>
#include <fstream>
#include <iostream>
#include <boost/range/combine.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/lambda/bind.hpp>
#include <GadgetronTimer.h>
#include <mri_core_coil_map_estimation.h>
//#include <generic_recon_gadgets/GenericReconBase.h>
#include <boost/hana/functional/iterate.hpp>
#include <cuNDArray_converter.h>
#include <cuNFFT.h>
#include <cuNDFFT.h>
#include <cuNDArray_math.h>
#include <cuNDArray.h>
#include <b1_map.h>
#include <cudaDeviceManager.h>
#include <iterator>
#include <SpiralBuffer.h>
#include <omp.h>
#include <mri_core_kspace_filter.h>
#include <vector_td_utilities.h>
#include <cuSDC.h>
#include <ismrmrd/xml.h>
#include "reconParams.h"

#include "mri_core_girf_correction_lit.h"

using namespace Gadgetron;
using namespace Gadgetron::Core;
//using namespace Gadgetron::Core
namespace nhlbi_toolbox
{
        namespace utils
        {
                cuNDArray<float_complext> estimateCoilmaps_slice(cuNDArray<float_complext> &data);
                
                void attachHeadertoImageArray(ImageArray &imarray, ISMRMRD::AcquisitionHeader acqhdr, const ISMRMRD::IsmrmrdHeader &h);
                void attachHeadertoImageArrayReconParams(ImageArray &imarray, ISMRMRD::AcquisitionHeader acqhdr, Gadgetron::reconParams &recon_params);
        
                std::tuple<std::vector<std::vector<size_t>>,hoNDArray<size_t>> sort_idx_phases(std::vector<std::vector<std::vector<std::vector<size_t>>>> idx_phases_vec,std::vector<size_t> ordering, bool collapse_to_last,float start_index_nc=0.0);

                void filterImagealongSlice(hoNDArray<std::complex<float>> &image, ISMRMRDKSPACEFILTER ftype, size_t fwidth, double fsigma);
                template <size_t D=3> void filterImage(hoNDArray<std::complex<float>> &image, ISMRMRDKSPACEFILTER ftype, size_t fwidth, double fsigma);
                int selectCudaDevice();
                std::vector<int> FindCudaDevices(unsigned long);
                void setNumthreadsonGPU(int Number);
                template <typename T>
                void write_gpu_nd_array(cuNDArray<T> &data, std::string filename);
                template <typename T>
                void write_cpu_nd_array(hoNDArray<T> &data, std::string filename);
                template <typename T>
                cuNDArray<T> concat(std::vector<cuNDArray<T>> &arrays);
                template <typename T>
                hoNDArray<T> concat(std::vector<hoNDArray<T>> &arrays);
                float correlation(hoNDArray<float> a, hoNDArray<float> b);

                template <typename T>
                std::vector<T> sliceVec(std::vector<T> &v, int start, int end, int stride);

                template <typename T>
                std::vector<size_t> sort_indexes(std::vector<T> &v);

                template <typename T>
                void normalizeImages(hoNDArray<T> &input_image);

                template <typename T>
                hoNDArray<T> padForConv(hoNDArray<T> &input);

                template <typename T>
                hoNDArray<T> convolve(hoNDArray<T> &input, hoNDArray<T> &kernel);

                template <typename T>
                hoNDArray<T> paddedCovolution(hoNDArray<T> &input, hoNDArray<T> &kernel);

                void normalize_trajectory(hoNDArray<floatd2> *traj_input);

                std::vector<hoNDArray<float>> estimateDCF_slice(std::vector<std::vector<hoNDArray<floatd3>>> trajectories, std::vector<std::vector<hoNDArray<float>>> dcf, bool useIterativeDCWEstimated, float oversampling_factor_, size_t iterations,
                                                                std::vector<size_t> image_dims_, bool fullySampled);

                template <template <class> class ARRAY>
                void set_data(ARRAY<float_complext> &sp_data, ARRAY<floatd3> &sp_traj, ARRAY<float> &sp_dcw, std::vector<ISMRMRD::AcquisitionHeader> &sp_headers,
                              boost::shared_ptr<cuNDArray<float_complext>> &cuData, boost::shared_ptr<cuNDArray<floatd3>> &traj, boost::shared_ptr<cuNDArray<float>> &dcw, std::vector<size_t> indata_dims, std::vector<size_t> outdata_dims, int currDev);

                arma::fmat33 lookup_PCS_DCS(std::string PCS_description);

                hoNDArray<float> estimateDCF(hoNDArray<floatd3> trajectories, hoNDArray<float> dcf, bool useIterativeDCWEstimated, float oversampling_factor_, size_t iterations,
                                             std::vector<size_t> image_dims_, bool fullySampled);

                std::vector<arma::uvec> extractSlices(hoNDArray<floatd3> sp_traj);

                template <class T>
                hoNDArray<T> mean_complex(hoNDArray<T> input,unsigned int dim);

                template <class T>
                hoNDArray<T> std_complex(hoNDArray<T> input,unsigned int dim);

                template <class T>
                hoNDArray<T> std_real(hoNDArray<T> input,unsigned int dim);

                hoNDArray<floatd3> traj2grad(const hoNDArray<floatd3> &trajectory, float kspace_scaling, ISMRMRD::AcquisitionHeader head);
                hoNDArray<floatd2> traj2grad(const hoNDArray<floatd2> &trajectory, float kspace_scaling, ISMRMRD::AcquisitionHeader head);
                hoNDArray<floatd2> traj2grad_3D2D(const hoNDArray<floatd3> &trajectory, float kspace_scaling, ISMRMRD::AcquisitionHeader head);

                std::vector<size_t> linspace(size_t a, size_t b, size_t num);

                template <typename T>
                cuNDArray<T> set_device(cuNDArray<T> *, int device);

                std::tuple<boost::shared_ptr<hoNDArray<floatd3>>, boost::shared_ptr<hoNDArray<float>>> separate_traj_and_dcw_3D_gen(
                    hoNDArray<floatd2> *traj_input, hoNDArray<float> *dcw_input, int iSL, float enc_mat_z);

                std::tuple<boost::shared_ptr<hoNDArray<floatd2>>, boost::shared_ptr<hoNDArray<float>>> separate_traj_and_dcw_2(
                    hoNDArray<float> *traj_dcw, float enc_mat_z);

                std::tuple<boost::shared_ptr<hoNDArray<floatd3>>, boost::shared_ptr<hoNDArray<float>>> separate_traj_and_dcw_3D(
                    hoNDArray<float> *traj_dcw, int iSL, float enc_mat_z);
                template<unsigned int D>
        std::tuple<boost::shared_ptr<hoNDArray<vector_td<float,D>>>, boost::shared_ptr<hoNDArray<float>>> separate_traj_and_dcw_2or3(
            hoNDArray<float> *traj_dcw);

                cuNDArray<float> padDeformations(cuNDArray<float> deformation, std::vector<size_t> size_deformation);


                constexpr double GAMMA = 4258.0; /* Hz/G */
                void enable_peeraccess();

        }
} // namespace utils
