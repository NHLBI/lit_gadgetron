/*
  An example of how to estimate DCF
*/
#pragma once
// Gadgetron includes
#include <cuSDC.h>
#include <hoNDArray_utils.h>

#include <cuNFFT.h>
#include <cuNDFFT.h>
#include <cuNDArray_math.h>
#include <cuNDArray.h>
#include <cuNDArray_math.h>
#include <cuNDArray_operators.h>
#include <cuNonCartesianSenseOperator.h>
#include <cuCgSolver.h>
#include <cuNlcgSolver.h>
#include <cuCgPreconditioner.h>
#include <cuImageOperator.h>
#include <cuTvOperator.h>
#include <cuTvPicsOperator.h>
#include <cuNlcgSolver.h>
#include <cuPartialDerivativeOperator.h>
#include <cuSbcCgSolver.h>

#include <hoNDArray_fileio.h>
#include <parameterparser.h>
#include <NFFTOperator.h>

// Std includes
#include <iostream>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <GadgetronTimer.h>
#include <boost/program_options.hpp>
#include <boost/asio.hpp>
#include <mri_core_coil_map_estimation.h>
#include <hoArmadillo.h>
#include "util_functions.h"
#include "noncartesian_reconstruction.h"
#include "noncartesian_reconstruction_4D.h"
#include "noncartesian_reconstruction_5D.h"
#include "noncartesian_reconstruction_3D.h"
#include "reconParams.h"
#include "densityCompensation.h"
#include <complex>

// #include <python_toolbox.h>

using namespace Gadgetron;
using namespace Gadgetron::Core;
using namespace std;

// Define desired precision
typedef float _real;
namespace po = boost::program_options;
uint64d3 image_dims_os_;
std::vector<size_t> image_dims_;
float kernel_width_;
boost::shared_ptr<cuNFFT_plan<float, 3>> nfft_plan_;
boost::shared_ptr<cuNDArray<float_complext>> cuData;

int main(int argc, char **argv)
{
    std::vector<size_t> recon_dims;

    //
    // Parse command line
    //
    std::string trajectory_file;
    std::string def_file;
    std::string invdef_file;
    std::string data_file;
    std::string dcw_file;
    std::string csm_file;
    std::string out_file;
    std::string recondim_file;
    std::string erecondim_file;
    std::string fov_file;
    std::string out_file_csm;
    std::string shotspertime_file;
    std::string combination_weights_file;
    std::string scaled_time_file;
    std::string fbins_file;

    float oversampling_factor_;
    float lambda_spatial;
    float lambda_time;
    float lambda_time2;
    float tolerance;
    float lF;
    float uF;

    // dcf
    float kernel_width_dcf;
    float iterations_dcf;
    float oversampling_factor_dcf;
    bool save_dcf;
    bool calculateDCF;
    bool calculateKPRECOND;
    bool calculateCSM;
    bool isCI;
    bool bstar;
    int norm;
    bool use_gcc;
    int iterations_;

    bool pseudoReplica;
    bool justChannelImages = false;
    size_t xsize_;
    size_t ysize_;
    size_t ezsize_;
    size_t rzsize_;
    size_t numFbins;
    size_t reconstructionType; // 0-csm + recon, 1-cgsense, 2-3DTV, 3-4DspatialTempTV
    size_t gcc_coils;

    size_t csmType;
    // Initialize to prevent errors
    lF = 0;
    uF = 0;
    numFbins = 1;
    norm = 1;
    pseudoReplica = false;
    bstar = false;
    use_gcc = false;
    po::options_description desc("Allowed options");
    desc.add_options()("help,h", "Produce help message")                                                          //
        ("trajectory,t", po::value<std::string>(&trajectory_file), "trajectory file to read trajectory")          //
        ("shortPerTime,spt", po::value<std::string>(&shotspertime_file), "shots per time")                        //
        ("data,d", po::value<std::string>(&data_file), "data file to read data")                                  //
        ("dcw,w", po::value<std::string>(&dcw_file), "dcw file to read dcw")                                      //
        ("csm,c", po::value<std::string>(&csm_file), "csm file to read csm")                                      //
        ("combination_weights,cw", po::value<std::string>(&combination_weights_file), "combination_weights_file") //
        ("scaled_time,sct", po::value<std::string>(&scaled_time_file), "scaled_time_file")                        //
        ("fbins,fb", po::value<std::string>(&fbins_file), "fbins_file")                                           //
        ("out,o", po::value<std::string>(&out_file), "out file to write images")                                  //
        ("csmout,csmout", po::value<std::string>(&out_file_csm), "outputFileCSM")                                 //
        ("reconDims,rd", po::value<std::string>(&recondim_file), "recon dims")                                    //
        ("ereconDims,erd", po::value<std::string>(&erecondim_file), "erecon dims")                                //
        ("fov,fo", po::value<std::string>(&fov_file), "fov dims")                                                 //
        ("oversampling,f", po::value<float>(&oversampling_factor_)->default_value(2.1), "oversampling factor")                        //
        ("iterations,i", po::value<int>(&iterations_)->default_value(10), "size of reconst z")                                       //
        ("kwidth,k", po::value<float>(&kernel_width_)->default_value(3), "kernel width")                                            //
        ("kwidth_dcf,k_dcf", po::value<float>(&kernel_width_dcf)->default_value(5.5), "kernel width for dcf")                         //
        ("iterations_dcf,i_dcf", po::value<float>(&iterations_dcf)->default_value(10), "iterations for dcf")                         //
        ("oversampling_dcf,o_dcf", po::value<float>(&oversampling_factor_dcf)->default_value(2.1), "oversampling factor dcf")         //
        ("save_dcf,save_dcf", po::value<bool>(&save_dcf)->default_value(false), "true/false")                                           //
        ("calculateDCF,calc_dcf", po::value<bool>(&calculateDCF)->default_value(true), "Flag to calculate DCF")
        ("calculateKPRECOND", po::value<bool>(&calculateKPRECOND)->default_value(false), "Flag to calculate kprecond")                        //
        ("calculateCSM,s", po::value<bool>(&calculateCSM)->default_value(true), "Flag to calculate CSM")                               //
        ("tolerance,tol", po::value<float>(&tolerance)->default_value(200), "tolerance_sense")                                        //
        ("lambda_spatial,ls", po::value<float>(&lambda_spatial)->default_value(0.05), "lambda_spatial")                                //
        ("lambda_time,lt", po::value<float>(&lambda_time)->default_value(0.05), "lambda_time")                                         //
        ("lambda_time2,lt", po::value<float>(&lambda_time2)->default_value(0.05), "lambda_time2")                                      //
        ("deformation,def", po::value<std::string>(&def_file), "deformation field")                               //
        ("invdeformation,idef", po::value<std::string>(&invdef_file), "inv deformation field")                    //
        ("bstar,bstar", po::value<bool>(&bstar)->default_value(false), "true/false")                                                    //
        ("norm,norm", po::value<int>(&norm)->default_value(2), "1/2")                                                               //
        ("justChannelImages,jci", po::value<bool>(&justChannelImages)->default_value(false), "justChannelImages")                       //
        ("reconstructionType,rt", po::value<size_t>(&reconstructionType)->default_value(0), "reconstructionType: 0-csm + gridding recon, 1-cgsense, 2-3DTV, 3-4DspatialTempTV, 4- concom Sense spTV3D, 5- concom 4DspatialTempTV, 6- icomoco_fc, 7-4DspatialTempTV_nlcg  ")
        ("pseudoReplica,pr", po::value<bool>(&pseudoReplica)->default_value(false), "true/false")
        ("use_gcc,gcc", po::value<bool>(&use_gcc)->default_value(false), "true/false")
        ("csmType,csmType", po::value<size_t>(&csmType)->default_value(0), "Roemer 0 / Espirit 1")
        ("gcc_coils,gcc_coils", po::value<size_t>(&gcc_coils)->default_value(6), "number of gcc coils");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    
    if (vm.count("help"))
    {
        std::cout << desc << std::endl;
        return 0;
    }

    auto eligibleGPUs = nhlbi_toolbox::utils::FindCudaDevices(0);

    int selectedDevice = nhlbi_toolbox::utils::selectCudaDevice();

    if (std::find(eligibleGPUs.begin(), eligibleGPUs.end(), selectedDevice) == eligibleGPUs.end())
        selectedDevice = eligibleGPUs[0];

    GDEBUG_STREAM("Selected Device: " << selectedDevice);
    cudaSetDevice(selectedDevice);

    unsigned int warp_size = cudaDeviceManager::Instance()->warp_size();
    GDEBUG_STREAM("warp_size: " << warp_size);

    reconParams recon_params;

    boost::shared_ptr<hoNDArray<vector_td<float, 3>>> trajectory_in =
        read_nd_array<floatd3>((char *)trajectory_file.c_str());

    boost::shared_ptr<hoNDArray<float>> dcf_in =
        read_nd_array<float>((char *)dcw_file.c_str());

    boost::shared_ptr<hoNDArray<float>> recon_dims_read =
        read_nd_array<float>((char *)recondim_file.c_str());

    boost::shared_ptr<hoNDArray<float>> fov_file_read =
        read_nd_array<float>((char *)fov_file.c_str());

    boost::shared_ptr<hoNDArray<float>> erecon_dims_read =
        read_nd_array<float>((char *)erecondim_file.c_str());

    boost::shared_ptr<hoNDArray<float_complext>> csm_in =
        read_nd_array<float_complext>((char *)csm_file.c_str());

    boost::shared_ptr<hoNDArray<float>> def_in =
        read_nd_array<float>((char *)def_file.c_str());

    boost::shared_ptr<hoNDArray<float>> invdef_in =
        read_nd_array<float>((char *)invdef_file.c_str());

    hoNDArray<size_t> shots_per_time =
        hoNDArray<size_t>(*read_nd_array<float>((char *)shotspertime_file.c_str()));

    hoNDArray<float_complext> hocw =
        hoNDArray<float_complext>(*read_nd_array<float_complext>((char *)combination_weights_file.c_str()));

    hoNDArray<float> hosct =
        hoNDArray<float>(*read_nd_array<float>((char *)scaled_time_file.c_str()));

    hoNDArray<float> hofbins =
        hoNDArray<float>(*read_nd_array<float>((char *)fbins_file.c_str()));

    boost::shared_ptr<hoNDArray<float_complext>> data_in =
        read_nd_array<float_complext>((char *)data_file.c_str());

    std::vector<float> mat_size(recon_dims_read.get()->get_data_ptr(), recon_dims_read.get()->get_data_ptr() + recon_dims_read.get()->get_number_of_elements());
    std::vector<float> emat_size(erecon_dims_read.get()->get_data_ptr(), erecon_dims_read.get()->get_data_ptr() + recon_dims_read.get()->get_number_of_elements());
    std::vector<float> fov_mat(fov_file_read.get()->get_data_ptr(), fov_file_read.get()->get_data_ptr() + fov_file_read.get()->get_number_of_elements());
    // hoNDArray<size_t> shots_per_time(shotspertime_read.get_data_ptr(), shotspertime_read.get_data_ptr() + shotspertime_read.get_number_of_elements());
    std::vector<float> fbins_vec(hofbins.get_data_ptr(), hofbins.get_data_ptr() + hofbins.get_number_of_elements());

    arma::fvec fbins(fbins_vec);

    ISMRMRD::MatrixSize ematsize, rmatsize;
    ISMRMRD::FieldOfView_mm fov;
    if (size_t(emat_size[0]) % 32 != 0)
        GDEBUG_STREAM("Please select image matrices dim 0 to be multiples of 32");
    if (size_t(emat_size[1]) % 32 != 0)
        GDEBUG_STREAM("Please select image matrices dim 1 to be multiples of 32");
    if (size_t(emat_size[2]) % 32 != 0)
        GDEBUG_STREAM("Please select image matrices dim 2 to be multiples of 32");

    auto factor_0 = float((size_t(emat_size[0]) / 32) * 32) / float(size_t(emat_size[0]));
    auto factor_1 = float((size_t(emat_size[1]) / 32) * 32) / float(size_t(emat_size[1]));
    auto factor_2 = float((size_t(emat_size[2]) / 32) * 32) / float(size_t(emat_size[2]));

    if (bstar)
        factor_2 = factor_2 == 0 ? 1 : factor_2;
    else
        factor_2 = 1;

    ematsize.x = size_t(emat_size[0] * factor_0);
    ematsize.y = size_t(emat_size[1] * factor_1);
    ematsize.z = size_t(emat_size[2] * factor_2);

    auto factor_0r = float((size_t(mat_size[0]) / 32) * 32) / float(size_t(mat_size[0]));
    auto factor_1r = float((size_t(mat_size[1]) / 32) * 32) / float(size_t(mat_size[1]));
    auto factor_2r = float((size_t(mat_size[2]) / 32) * 32) / float(size_t(mat_size[2]));

    if (bstar)
        factor_2r = factor_2r == 0 ? 1 : factor_2r;
    else
        factor_2r = 1;

    rmatsize.x = size_t(mat_size[0] * factor_0r);
    rmatsize.y = size_t(mat_size[1] * factor_1r);
    rmatsize.z = size_t(mat_size[2] * factor_2r);

    fov.x = fov_mat[0];
    fov.y = fov_mat[1];
    fov.z = fov_mat[2];

    GDEBUG_STREAM("Factor Encoded: X " << factor_0 << " Y " << factor_1 << " Z " << factor_2);
    GDEBUG_STREAM("Factor Recon: X " << factor_0r << " Y " << factor_1r << " Z " << factor_2r);

    GDEBUG_STREAM("Encoded Matrix: X " << ematsize.x << " Y " << ematsize.y << " Z " << ematsize.z);
    GDEBUG_STREAM("Recon Matrix: X " << rmatsize.x << " Y " << rmatsize.y << " Z " << rmatsize.z);
    
    ISMRMRD::MatrixSize omatsize;
    omatsize.x=size_t(ematsize.x);
    omatsize.y=size_t(ematsize.y);
    omatsize.z=size_t(ematsize.z);
    if(calculateKPRECOND){
        GDEBUG_STREAM("WARNING Kspace Preconditioner is not working with enlarge FOV !!!");
    } 

    recon_params.omatrixSize = omatsize;
    recon_params.ematrixSize = ematsize;
    recon_params.rmatrixSize = rmatsize;
    recon_params.fov = fov;
    recon_params.shots_per_time = shots_per_time;
    recon_params.oversampling_factor_ = oversampling_factor_;
    recon_params.kernel_width_ = kernel_width_;
    recon_params.iterations = iterations_;
    recon_params.tolerance = tolerance;
    recon_params.numberChannels = data_in.get()->get_size(data_in.get()->get_number_of_dimensions() - 1);
    recon_params.selectedDevice = selectedDevice;
    recon_params.selectedDevices = {selectedDevice,eligibleGPUs[1]}; // TO FIX quick workaround
    recon_params.lambda_spatial = lambda_spatial;
    recon_params.lambda_time2 = lambda_time2;
    recon_params.lambda_spatial_imoco = lambda_spatial;
    recon_params.iterations_imoco = iterations_;
    recon_params.lambda_time = lambda_time;
    recon_params.kernel_width_dcf_ = kernel_width_dcf;
    recon_params.iterations_dcf = iterations_dcf;
    recon_params.oversampling_factor_dcf_ = oversampling_factor_dcf;
    recon_params.use_gcc = use_gcc;
    recon_params.RO = data_in.get()->get_size(0);
    recon_params.norm = norm;
    recon_params.gcc_coils = gcc_coils;
    recon_params.try_channel_gridding = false;
    // GDEBUG_STREAM("def[0]: " << def_in.get()->get_size(0));
    // GDEBUG_STREAM("def[1]: " << def_in.get()->get_size(1));
    // GDEBUG_STREAM("def[2]: " << def_in.get()->get_size(2));
    // GDEBUG_STREAM("def[3]: " << def_in.get()->get_size(3));
    // GDEBUG_STREAM("def[4]: " << def_in.get()->get_size(4));
    // GDEBUG_STREAM("def[5]: " << def_in.get()->get_size(5));

    // GDEBUG_STREAM("csm[0]: " << csm_in.get()->get_size(0));
    // GDEBUG_STREAM("csm[1]: " << csm_in.get()->get_size(1));
    // GDEBUG_STREAM("csm[2]: " << csm_in.get()->get_size(2));
    // GDEBUG_STREAM("csm[3]: " << csm_in.get()->get_size(3));
    // GDEBUG_STREAM("data_in[3]: " << data_in.get()->get_size(2));

    // GDEBUG_STREAM("csm[0]: " << csm_in.get()->get_size(0));
    // GDEBUG_STREAM("csm[1]: " << csm_in.get()->get_size(1));
    // GDEBUG_STREAM("csm[2]: " << csm_in.get()->get_size(2));
    // GDEBUG_STREAM("csm[3]: " << csm_in.get()->get_size(3));
    // GDEBUG_STREAM("data_in[3]: " << data_in.get()->get_size(2));
    boost::shared_ptr<Gadgetron::cuNDArray<Gadgetron::float_complext>> csm;
    cuNDArray<Gadgetron::float_complext> images;

    nhlbi_toolbox::reconstruction::noncartesian_reconstruction<3> reconstruction(recon_params);
    nhlbi_toolbox::reconstruction::noncartesian_reconstruction_3D reconstruction3D(recon_params);
    nhlbi_toolbox::reconstruction::noncartesian_reconstruction_4D reconstruction4D(recon_params);
    nhlbi_toolbox::reconstruction::noncartesian_reconstruction_5D reconstruction5D(recon_params);

    cuNDArray<float> dcf_o;

    std::vector<cuNDArray<float>> scaled_time_vec;
    cuNDArray<float_complext> cw;
    cuNDArray<float> sct;

    GDEBUG_STREAM("Input data shape: RO " << data_in->get_size(0) <<" INT "<< data_in->get_size(1) << " CHA " << data_in->get_size(2));
    GDEBUG_STREAM("Input data  expected shape: RO " << recon_params.RO << " CHA " << recon_params.numberChannels);
    if (recon_params.numberChannels!=data_in->get_size(2)){
        *data_in=permute(*data_in,{0,2,1});
         GDEBUG_STREAM("Input data Permuted");   
    }


    if (reconstructionType == 4 || reconstructionType == 5 || reconstructionType == 6)
    {
        cw = cuNDArray<float_complext>(hocw);
        sct = cuNDArray<float>(hosct);
        for (size_t it = 0; it < shots_per_time.get_number_of_elements(); it++)
        {
            std::vector<size_t> sct_dims = {recon_params.RO, *(shots_per_time.begin() + it)};
            size_t inter_acc = std::accumulate(shots_per_time.begin(), shots_per_time.begin() + it, 0) * sct.get_size(0);
            if (sct.get_size(0) < 1)
            {
                cuNDArray<float> temp(sct_dims);
                fill(&temp, float(0.0));
                scaled_time_vec.push_back(temp);
            }
            else
            {
                auto temp = cuNDArray<float>(sct_dims, sct.data() + inter_acc);
                scaled_time_vec.push_back(temp);
            }
        }
    }

    if (!calculateCSM)
    {

        GDEBUG_STREAM("Received CSM ");
        if (csm_in.get()->get_size(0) == data_in.get()->get_size(2))
            *csm_in = permute(*csm_in, {3, 2, 1, 0});
        GDEBUG_STREAM("CSM shape: X " << csm_in.get()->get_size(0) << " Y " << csm_in.get()->get_size(1) << " Z " << csm_in.get()->get_size(2) << " CHA " << csm_in.get()->get_size(3));
        GDEBUG_STREAM("Expected CSM shape X" << rmatsize.x << "Y " << rmatsize.y << " Z " << reconstruction.get_recon_dims()[2] << " CHA " << data_in.get()->get_size(2));
        csm = boost::make_shared<cuNDArray<float_complext>>(pad<float_complext, 4>(uint64d4(ematsize.x, ematsize.y, reconstruction.get_recon_dims()[2], csm_in->get_size(3)),
                                                                                   *csm_in, float_complext(0)));
        // csm = boost::make_shared<cuNDArray<float_complext>>(*csm_in);
    }

    if (pseudoReplica)
    {
        GDEBUG_STREAM("SNR Pseudo replica is not implemented");
    }

    // reconstructionType 0 : Average Image
    if (calculateCSM || reconstructionType == 0 || justChannelImages || use_gcc)
    {
        GadgetronTimer timer("Reconstruct");
        auto [cuData, traj_csm, dcf_csm] = reconstruction3D.organize_data(data_in.get(), trajectory_in.get(), dcf_in.get(), calculateDCF, calculateKPRECOND);
        if (save_dcf && reconstructionType == 0)
        {
            dcf_o = dcf_csm;
        }
        square_inplace(&dcf_csm);
        cuNDArray<float_complext> channel_images(reconstruction.get_recon_dims());
        {
            GadgetronTimer timer("Reconstruct OG");
            reconstruction.reconstruct(&cuData, &channel_images, &traj_csm, &dcf_csm);
        }
        if (recon_params.use_gcc)
        {
            GadgetronTimer timer("GCC Compression");
            channel_images = reconstruction.estimate_gcc_matrix(channel_images);
        }
        if (justChannelImages)
        {
            GadgetronTimer timer("Write Channel images");
            nhlbi_toolbox::utils::write_gpu_nd_array<float_complext>(channel_images, (out_file + std::string("_channelImages.complex")));
        }
        cuData.clear();
        traj_csm.clear();
        dcf_csm.clear();

        if (calculateCSM)
        {
            switch (csmType)
            {
                case 0:
                {
                    GDEBUG_STREAM("Calculating Roemer CSM");
                    csm = reconstruction.generateRoemerCSM(&channel_images);
                }
                break;

                case 1:
                {
                    GDEBUG_STREAM("Calculating Espirit CSM");
                    csm = reconstruction.generateEspiritCSM(&channel_images);
                }
            break;
            }
            
            GDEBUG_STREAM("Writing CSM");
            nhlbi_toolbox::utils::write_gpu_nd_array<float_complext>(*csm, (out_file + std::string("_csm.complex")));
        }
        channel_images *= *conj(csm.get());
        auto combined = sum(&channel_images, channel_images.get_number_of_dimensions() - 1);
        cuNDArray<float_complext> ci_cropped = reconstruction.crop_to_recondims(*combined);
        if (reconstructionType == 0)
        {
            images = ci_cropped;
        }
        else
        {
            nhlbi_toolbox::utils::write_gpu_nd_array<float_complext>(ci_cropped, (out_file + std::string("_combined.complex")));
        }
        if (justChannelImages)
        {
            images = channel_images;
        }
        channel_images.clear();
        (*combined).clear();
    }
    
    switch (reconstructionType)
    {
    case 1:
    {
        auto [cuData, traj, dcw] = reconstruction3D.organize_data(data_in.get(), trajectory_in.get(), dcf_in.get(), calculateDCF, calculateKPRECOND);
        images = reconstruction3D.reconstruct_CGSense(&cuData, &traj, &dcw, csm);
        if (save_dcf)
        {
            dcf_o = dcw;
        }
    }
    break;

    case 2:
    {
        auto [cuData, traj, dcw] = reconstruction3D.organize_data(data_in.get(), trajectory_in.get(), dcf_in.get(), calculateDCF, calculateKPRECOND);
        images = reconstruction3D.reconstruct(&cuData, &traj, &dcw, csm);
        if (save_dcf)
        {
            dcf_o = dcw;
        }
    }
    break;
    case 3:
    {

        auto [cuData, trajVec, dcwVec] = reconstruction4D.organize_data(data_in.get(), trajectory_in.get(), dcf_in.get(), calculateDCF, calculateKPRECOND);

        if (recon_params.use_gcc)
        {
            auto mtx_vec = reconstruction.get_mtx_vec();
            cuData = reconstruction4D.apply_coil_compression(&cuData, &trajVec, &dcwVec, &mtx_vec);
            images = reconstruction4D.reconstruct(&cuData, &trajVec, &dcwVec, csm);
        }
        else
            images = reconstruction4D.reconstruct(&cuData, &trajVec, &dcwVec, csm);

        if (save_dcf)
        {
            dcf_o = reconstruction.vectortoarray(&dcwVec);
        }
    }
    break;
    case 4:
    {
        auto [cuData, traj, dcw] = reconstruction3D.organize_data(data_in.get(), trajectory_in.get(), dcf_in.get(), calculateDCF, calculateKPRECOND);
        images = reconstruction3D.reconstruct_CGSense_fc(&cuData, &traj, &dcw, csm, &cw, &sct, fbins);
        if (save_dcf)
        {
            dcf_o = dcw;
        }
    }
    break;
    case 5:
    {
        auto [cuData, trajVec, dcwVec] = reconstruction4D.organize_data(data_in.get(), trajectory_in.get(), dcf_in.get(), calculateDCF, calculateKPRECOND);
        images = reconstruction4D.reconstruct_fc(&cuData, &trajVec, &dcwVec, csm, &cw, scaled_time_vec, fbins);
        if (save_dcf)
        {
            dcf_o = reconstruction.vectortoarray(&dcwVec);
        }
    }
    break;
    case 6:
    {
        auto def = cuNDArray<float>(*def_in);
        auto invdef = cuNDArray<float>(*invdef_in);
        auto [cuData, trajVec, dcwVec] = reconstruction4D.organize_data(data_in.get(), trajectory_in.get(), dcf_in.get(), calculateDCF, calculateKPRECOND);

        images = reconstruction4D.reconstructiMOCO_fc(&cuData, &trajVec, &dcwVec, csm, &cw, scaled_time_vec, fbins);

        if (save_dcf)
        {
            dcf_o = reconstruction.vectortoarray(&dcwVec);
        }
    }
    break;
    case 7:
    {
        auto [cuData, trajVec, dcwVec] = reconstruction4D.organize_data(data_in.get(), trajectory_in.get(), dcf_in.get(), calculateDCF, calculateKPRECOND);
        images = reconstruction4D.reconstruct_nlcg(&cuData, &trajVec, &dcwVec, csm);
        if (save_dcf)
        {
            dcf_o = reconstruction.vectortoarray(&dcwVec);
        }
    }
    break;
    case 9:
    {

        auto [cuData, trajVec, dcwVec] = reconstruction4D.organize_data(data_in.get(), trajectory_in.get(), dcf_in.get(), calculateDCF, calculateKPRECOND);

        int deformation_gpu = nhlbi_toolbox::utils::selectCudaDevice();
        GDEBUG_STREAM("deformation_gpu" << deformation_gpu);
        cudaSetDevice(deformation_gpu);
        auto def = cuNDArray<float>(*def_in);
        auto invdef = cuNDArray<float>(*invdef_in);
        cudaSetDevice(cuData.get_device());
        images = reconstruction4D.reconstructiMOCO(&cuData, &trajVec, &dcwVec, csm);
        if (save_dcf)
        {
            dcf_o = reconstruction.vectortoarray(&dcwVec);
        }
    }
    break;
    case 10:
    {

        auto [cuData, trajVec, dcwVec] = reconstruction5D.organize_data(data_in.get(), trajectory_in.get(), dcf_in.get(), calculateDCF, calculateKPRECOND);
        int deformation_gpu = nhlbi_toolbox::utils::selectCudaDevice();
        GDEBUG_STREAM("deformation_gpu" << deformation_gpu);
        cudaSetDevice(deformation_gpu);

        auto def = cuNDArray<float>(*def_in);
        auto invdef = cuNDArray<float>(*invdef_in);
        cudaSetDevice(cuData.get_device());
        images = reconstruction5D.reconstructiMOCO_withdef(&cuData, &trajVec, &dcwVec, csm,&def,&invdef);
        
        if (save_dcf)
        {
            dcf_o = reconstruction.vectortoarray(&dcwVec);
        }
    }
    break;

    case 11:
    {

        auto [cuData, trajVec, dcwVec] = reconstruction5D.organize_data(data_in.get(), trajectory_in.get(), dcf_in.get(), calculateDCF, calculateKPRECOND);
        auto trajIn = reconstruction.vectortoarray(&trajVec);
        
        /*
        int deformation_gpu = nhlbi_toolbox::utils::selectCudaDevice();
        GDEBUG_STREAM("deformation_gpu" << deformation_gpu);
        cudaSetDevice(deformation_gpu);

        auto def = cuNDArray<float>(*def_in);
        auto invdef = cuNDArray<float>(*invdef_in);
        cudaSetDevice(cuData.get_device());
        GDEBUG_STREAM("SHOTS_PER_TIME  X0" << shots_per_time.get_size(0) << " X1 " << shots_per_time.get_size(1));
        for (size_t it = 0; it < shots_per_time.get_number_of_elements(); it++) {
            GDEBUG_STREAM("it " << it << "SHOTs " << *(shots_per_time.begin() + it));
            auto result=float(dcwVec[it].get_number_of_elements())/float(recon_params.RO);
            GDEBUG_STREAM("dcw" << it << "SHOTs " << result);
            }

        //images = reconstruction5D.reconstructiMOCO_withdef(&cuData, &trajVec, &dcwVec, csm,&def,&invdef);
        */
        if (save_dcf)
        {
            dcf_o = reconstruction.estimate_dcf_special(&trajIn);
        }
        
    }
    break;


    }
    if (save_dcf)
    {
        GDEBUG_STREAM("Writing DCF");
        nhlbi_toolbox::utils::write_gpu_nd_array<float>(dcf_o, out_file + std::string("_dcf.real"));
    }

    GDEBUG_STREAM("Writing Image");
    nhlbi_toolbox::utils::write_gpu_nd_array<float_complext>(images, out_file + std::string("_images.complex"));
    GDEBUG_STREAM("Finished Recon");


    std::exit(0);
    return 0;
}