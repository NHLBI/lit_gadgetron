
set(gadgetron_nhlbi_gt_gadgets_header_files
    waveforms/WaveformToTrajectory.h
    waveforms/AcquisitionWaveformFanout.h    
    utility_gadgets/SpiralBufferFanout.h 
    utility_gadgets/AcquisitionReconParamsFanout.h 
)
set(gadgetron_nhlbi_gt_gadgets_src_files
noncart_recon/Noncart_recon_gadget.cpp
utility_gadgets/RemoveSpiralOversampling.cpp
utility_gadgets/ImagetoVector.cpp
waveforms/WaveformToTrajectory.cpp
waveforms/RemoveNavsGadget.cpp
utility_gadgets/PrepreconParams.cpp
utility_gadgets/SelectEchoes.cpp
utility_gadgets/AcquisitionReconParamsFanout.cpp
utility_gadgets/AcquisitionStopTimingGadget.cpp
waveforms/AcquisitionWaveformFanout.cpp
#waveforms/RemoveAcqsGadget.cpp
)

add_library(nhlbi_gt_gadgets SHARED
     ${gadgetron_nhlbi_gt_gadgets_header_files}
     ${gadgetron_nhlbi_gt_gadgets_src_files}
    )

set_target_properties(nhlbi_gt_gadgets PROPERTIES VERSION ${GADGETRON_VERSION_STRING} SOVERSION ${GADGETRON_SOVERSION})

target_link_libraries(nhlbi_gt_gadgets
        gadgetron_core
        gadgetron_toolbox_cpucore_math
        gadgetron_toolbox_cpunfft
        gadgetron_toolbox_cpureg
        gadgetron_mricore
        gadgetron_toolbox_cpufft
        gadgetron_toolbox_mri_core
        gadgetron_toolbox_log
        gadgetron_toolbox_gpu
        gadgetron_toolbox_spiral
        gadgetron_toolbox_cpucore
        gadgetron_gpuparallelmri
        gadgetron_core_parallel
        nhlbi_gt_toolbox
        
        )

target_include_directories(nhlbi_gt_gadgets
    PUBLIC
        $<INSTALL_INTERFACE:include>
        ${CMAKE_SOURCE_DIR}>
        ${CMAKE_SOURCE_DIR}/gadgets/mri_noncartesian>
        ${CMAKE_SOURCE_DIR}/core/parallel>
        ${CMAKE_SOURCE_DIR}/toolboxes/core/gpu>
        ${CMAKE_SOURCE_DIR}/toolboxes/gpu>
        ${CMAKE_SOURCE_DIR}/toolboxes/nfft/gpu>
        ${CMAKE_SOURCE_DIR}/toolboxes/nhlbi_gt_toolbox/utils>
        ${CMAKE_SOURCE_DIR}/toolboxes/nhlbi_gt_toolbox/gadgets>
        ${CMAKE_SOURCE_DIR}/toolboxes/nhlbi_gt_toolbox/utils/gpu>
        ${CMAKE_SOURCE_DIR}/toolboxes/nhlbi_gt_toolbox/gadgets/waveforms>
        ${CMAKE_SOURCE_DIR}/toolboxes/nhlbi_gt_toolbox/gadgets/utility_gadgets>
        ${CMAKE_SOURCE_DIR}/toolboxes/nhlbi_gt_toolbox/spiral>
        ${CMAKE_SOURCE_DIR}/toolboxes/nhlbi_gt_toolbox/spiral/reconstruction>
        ${CMAKE_SOURCE_DIR}/toolboxes/nhlbi_gt_toolbox/utils/concomitantFieldCorrection/>
        ${CMAKE_SOURCE_DIR}/toolboxes/nhlbi_gt_toolbox/utils/gpu/gpuregistration>
        ${CMAKE_SOURCE_DIR}/toolboxes/nhlbi_gt_toolbox/utils/gpu/gpuregistration/internal>
)
install(FILES ${gadgetron_nhlbi_gt_gadgets_header_files}
        DESTINATION ${GADGETRON_INSTALL_INCLUDE_PATH} COMPONENT main)

install(TARGETS nhlbi_gt_gadgets
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
	RUNTIME DESTINATION bin
	COMPONENT main
)
