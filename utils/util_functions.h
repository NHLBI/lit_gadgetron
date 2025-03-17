#pragma once
#include <gadgetron/mri_core_grappa.h>
#include <gadgetron/vector_td_utilities.h>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <gadgetron/cgSolver.h>
#include <gadgetron/hoNDArray_math.h>
#include <gadgetron/hoNDArray_utils.h>
#include <gadgetron/hoNDArray_elemwise.h>
#include <gadgetron/hoNFFT.h>
#include <gadgetron/hoNDFFT.h>
#include <numeric>
#include <random>
#include <gadgetron/NonCartesianTools.h>
#include <gadgetron/NFFTOperator.h>
#include <sstream>
#include <fstream>
#include <iostream>
#include <boost/range/combine.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/lambda/bind.hpp>
#include <gadgetron/GadgetronTimer.h>
#include <gadgetron/mri_core_coil_map_estimation.h>
//#include <gadgetron/generic_recon_gadgets/GenericReconBase.h>
#include <boost/hana/functional/iterate.hpp>
#include <gadgetron/cuNDArray_converter.h>
#include <gadgetron/cuNFFT.h>
#include <gadgetron/cuNDFFT.h>
#include <gadgetron/cuNDArray_math.h>
#include <gadgetron/cuNDArray.h>
#include <gadgetron/b1_map.h>
#include <gadgetron/cudaDeviceManager.h>
#include <iterator>
#include <omp.h>
#include <gadgetron/mri_core_kspace_filter.h>
#include <gadgetron/vector_td_utilities.h>
#include <gadgetron/cuSDC.h>
#include "gpu/cuda_utils.h"
#include <ismrmrd/xml.h>
#include <gadgetron/cuNDArray_fileio.h>
#include <gadgetron/mri_core_utility.h>

using namespace Gadgetron;
using namespace Gadgetron::Core;
//using namespace Gadgetron::Core
namespace nhlbi_toolbox
{
        namespace utils
        {
                cuNDArray<float_complext> estimateCoilmaps_slice(cuNDArray<float_complext> &data);
                
                void attachHeadertoImageArray(ImageArray &imarray, ISMRMRD::AcquisitionHeader acqhdr, const ISMRMRD::IsmrmrdHeader &h);

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

                arma::fmat33 lookup_PCS_DCS(std::string PCS_description);

                std::vector<arma::uvec> extractSlices(hoNDArray<floatd3> sp_traj);

                template <class T>
                hoNDArray<T> mean_complex(hoNDArray<T> input,unsigned int dim);

                template <class T>
                hoNDArray<T> std_complex(hoNDArray<T> input,unsigned int dim);

                template <class T>
                hoNDArray<T> std_real(hoNDArray<T> input,unsigned int dim);

                std::vector<size_t> linspace(size_t a, size_t b, size_t num);

                template <typename T>
                cuNDArray<T> set_device(cuNDArray<T> *, int device);

                /// chunk data N
                template <class T> void chunk_data_N(const hoNDArray<T>& data, bool chunk_N_bool, size_t chunk_N_input, hoNDArray<T>& res);
                /// chunk data S
                template <class T> void chunk_data_S(const hoNDArray<T>& data, bool chunk_S_bool, size_t chunk_S_input, hoNDArray<T>& res);

                constexpr double GAMMA = 4258.0; /* Hz/G */
                void enable_peeraccess();

        }
} // namespace utils
