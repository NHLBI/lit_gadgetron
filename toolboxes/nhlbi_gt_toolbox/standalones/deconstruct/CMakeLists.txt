###############################################################################################################################################################################

add_executable(deconstruction deconstruction.cpp)

target_link_libraries(deconstruction gadgetron_core gadgetron_toolbox_cpucore_math gadgetron_toolbox_spiral gadgetron_toolbox_cpunfft)

target_link_libraries(deconstruction
  gadgetron_toolbox_hostutils 
  gadgetron_toolbox_mri_core
  gadgetron_toolbox_cpusdc
	gadgetron_mri_noncartesian
  gadgetron_toolbox_spiral 
  gadgetron_toolbox_cpunfft
  gadgetron_toolbox_log
  gadgetron_toolbox_gpu
  ${CUDA_LIBRARIES}
  Boost::program_options
  nhlbi_gt_toolbox
  #Gadgetron_gpu
  #nhlbi_gt_utils
  gadgetron_toolbox_cpucore
  gadgetron_core
  Boost::filesystem 
  gadgetron_toolbox_python
  )
  
target_include_directories(deconstruction
        PUBLIC    
		$<INSTALL_INTERFACE:include> 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/gadgets/mri_noncartesian>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/core/parallel>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/toolboxes/core/gpu/>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/toolboxes/nfft/gpu/>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/nhlbi_gt_toolbox/>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/nhlbi_gt_toolbox/utils/>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/nhlbi_gt_toolbox/utils/gpu/>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/nhlbi_gt_toolbox/non_cartesian/>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/nhlbi_gt_toolbox/waveforms/>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/nhlbi_gt_toolbox/spiral/>
        ${CMAKE_PREFIX_PATH}/lib/
        ${CMAKE_PREFIX_PATH}/include/
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/../gadgets/mri_noncartesian>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/../gadgets/mri_core>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/nhlbi_gt_toolbox/utils/gpu/gpuregistration/>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/nhlbi_gt_toolbox/utils/gpu/gpuregistration/internal/>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/nhlbi_gt_toolbox/spiral/deconstruction/>
    ${CMAKE_SOURCE_DIR}/toolboxes/python/
    ${ISMRMRD_INCLUDE_DIR}
    ${CMAKE_PREFIX_PATH}/include/
    ${CMAKE_PREFIX_PATH}/include/
    ${CMAKE_PREFIX_PATH}/include/io/
)
install(TARGETS deconstruction DESTINATION bin COMPONENT main)


###############################################################################################################################################################################


