cmake_minimum_required(VERSION 3.17...3.21 FATAL_ERROR)

# project(NHLBI_GT_TOOLBOX LANGUAGES CXX C)

# set(GADGETRON_INSTALL_INCLUDE_PATH include/gadgetron)
# set(GADGETRON_INSTALL_CONFIG_PATH share/gadgetron/config)
#set(GADGETRON_INSTALL_PYTHON_MODULE_PATH share/python)

# Header and source files
set(gadgetron_nhlbi_gt_toolbox_header_files
    
    utils/mri_core_girf_correction_lit.h
    spiral/SpiralBuffer.h
    spiral/vds.h
    spiral/TrajectoryParameters_lit.h
    utils/concomitantFieldCorrection/mri_concomitant_field_correction_lit.h
    utils/util_functions.h
    utils/cpuSVD.h
    spiral/reconstruction/densityCompensation.h
    spiral/reconstruction/cuNonCartesianMOCOOperator_fc.h
    spiral/reconstruction/densityCompensation.h
    spiral/reconstruction/noncartesian_reconstruction.h
    spiral/reconstruction/reconParams.h
    spiral/reconstruction/SpiralReconBuffer.h
    spiral/reconstruction/noncartesian_reconstruction_2Dt.h
    spiral/reconstruction/noncartesian_reconstruction_4D.h
    spiral/reconstruction/noncartesian_reconstruction_5D.h
    spiral/reconstruction/noncartesian_reconstruction_3D.h
    spiral/reconstruction/noncartesian_reconstruction_fc.h
    spiral/reconstruction/cuNonCartesianSenseOperator_fc.h
    spiral/reconstruction/cuNonCartesianTSenseOperator_fc.h
    spiral/reconstruction/cuNonCartesianTSenseOperator.h
    spiral/reconstruction/cuNonCartesianMOCOOperator_fc.h
    spiral/reconstruction/cuNonCartesianMOCOOperator.h
    spiral/reconstruction/cuGpBbSolver_lit.h
    spiral/reconstruction/gpBbSolver_lit.h
    spiral/reconstruction/cuLowRankOperator.h
    spiral/reconstruction/cuLocallyLowRankOperator.h
    spiral/reconstruction/cuMOCOLowRankOperator.h
)

set(gadgetron_nhlbi_gt_toolbox_src_files
    utils/mri_core_girf_correction_lit.cpp
    spiral/TrajectoryParameters_lit.cpp
    spiral/vds.cpp
    spiral/reconstruction/noncartesian_reconstruction.cpp
    spiral/reconstruction/noncartesian_reconstruction_fc.cpp
    spiral/reconstruction/cuNonCartesianSenseOperator_fc.cpp
    spiral/reconstruction/cuNonCartesianTSenseOperator_fc.cpp
    spiral/reconstruction/cuNonCartesianTSenseOperator.cpp
    spiral/reconstruction/cuNonCartesianMOCOOperator_fc.cpp
    spiral/reconstruction/noncartesian_reconstruction_2Dt.cpp
    spiral/reconstruction/noncartesian_reconstruction_4D.cpp
    spiral/reconstruction/noncartesian_reconstruction_5D.cpp
    spiral/reconstruction/noncartesian_reconstruction_3D.cpp
    utils/util_functions.cpp 
    utils/cpuSVD.cpp 
    utils/concomitantFieldCorrection/mri_concomitant_field_correction_lit.cpp
    spiral/reconstruction/densityCompensation.cpp
    spiral/reconstruction/cuLocallyLowRankOperator.cpp
    spiral/reconstruction/cuLowRankOperator.cpp
    spiral/reconstruction/cuMOCOLowRankOperator.cpp
)

set(gadgetron_nhlbi_gt_toolbox_config_files
    config/pulmonary_MOCOLR.xml
    config/cardiovascular_iMOCO.xml
    config/cardiopulmonary_recon.xml
)


add_library(nhlbi_gt_toolbox SHARED
    ${gadgetron_nhlbi_gt_toolbox_header_files}
    ${gadgetron_nhlbi_gt_toolbox_src_files}
    ${gadgetron_nhlbi_gt_toolbox_config_files}
)

target_include_directories(nhlbi_gt_toolbox
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/gadgets/mri_noncartesian>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/core/parallel>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/toolboxes/core/gpu>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/toolboxes/gpu>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/toolboxes/nfft/gpu>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/toolboxes/nhlbi_gt_toolbox/utils>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/toolboxes/nhlbi_gt_toolbox/gadgets>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/toolboxes/nhlbi_gt_toolbox/utils/gpu>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/toolboxes/nhlbi_gt_toolbox/gadgets/waveforms>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/toolboxes/nhlbi_gt_toolbox/spiral>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/toolboxes/nhlbi_gt_toolbox/spiral/reconstruction>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/toolboxes/nhlbi_gt_toolbox/utils/concomitantFieldCorrection/>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/toolboxes/nhlbi_gt_toolbox/utils/gpu/gpuregistration>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/toolboxes/nhlbi_gt_toolbox/utils/gpu/gpuregistration/internal>
        ${CMAKE_CURRENT_SOURCE_DIR}/../fft/gpu
        ${CMAKE_CURRENT_SOURCE_DIR}/../core/gpu
        ${CMAKE_CURRENT_SOURCE_DIR}/../dwt/gpu
        ${CMAKE_CURRENT_SOURCE_DIR}/../fft/gpu
        ${CMAKE_CURRENT_SOURCE_DIR}/../mri/hyper
        ${CMAKE_CURRENT_SOURCE_DIR}/../mri/pmri/gpu
        ${CMAKE_CURRENT_SOURCE_DIR}/../nfft/gpu
        ${CMAKE_CURRENT_SOURCE_DIR}/../nfft/cpu
        ${CMAKE_CURRENT_SOURCE_DIR}/../operators/gpu
        ${CMAKE_CURRENT_SOURCE_DIR}/../registration/optical_flow
        ${CMAKE_CURRENT_SOURCE_DIR}/../registration/optical_flow/gpu
        ${CMAKE_CURRENT_SOURCE_DIR}/../mri/sdc/gpu
        ${CMAKE_CURRENT_SOURCE_DIR}/../solvers
        ${CMAKE_CURRENT_SOURCE_DIR}/../solvers/gpu
        ${CMAKE_PREFIX_PATH}/lib
        ${CMAKE_PREFIX_PATH}/include
)

target_link_libraries(nhlbi_gt_toolbox
    gadgetron_core
    gadgetron_toolbox_cpucore_math
    gadgetron_toolbox_cpunfft
    gadgetron_toolbox_cpureg
    gadgetron_mricore
    gadgetron_toolbox_cpufft
    gadgetron_toolbox_mri_core
    gadgetron_toolbox_log
    gadgetron_toolbox_gpu
)
set_target_properties(nhlbi_gt_toolbox PROPERTIES VERSION ${GADGETRON_VERSION_STRING} SOVERSION ${GADGETRON_SOVERSION})

install(FILES ${gadgetron_nhlbi_gt_toolbox_header_files} DESTINATION ${GADGETRON_INSTALL_INCLUDE_PATH} COMPONENT main)
install(FILES ${gadgetron_nhlbi_gt_toolbox_config_files} DESTINATION ${GADGETRON_INSTALL_CONFIG_PATH} COMPONENT main)

install(TARGETS nhlbi_gt_toolbox
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    COMPONENT main
)

add_subdirectory(utils)
#add_subdirectory(python)
add_subdirectory(gadgets)

add_subdirectory(standalones)
