cmake_minimum_required(VERSION 3.17...3.21 FATAL_ERROR)

if (${CMAKE_VERSION} VERSION_LESS 3.17)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
else ()
    cmake_policy(VERSION 3.17)
endif()
project(NHLBI_GT_TOOLBOX LANGUAGES CXX C)

find_program(CCACHE_FOUND ccache)
if (CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif (CCACHE_FOUND)

if (EXISTS $ENV{CMAKE_PREFIX_PATH})
    list(APPEND CMAKE_PREFIX_PATH $ENV{CMAKE_PREFIX_PATH})
endif ()
if (EXISTS $ENV{ISMRMRD_HOME})
    list(APPEND CMAKE_PREFIX_PATH $ENV{ISMRMRD_HOME})
endif ()
if (EXISTS $ENV{TORCH_HOME})
    list(APPEND CMAKE_PREFIX_PATH $ENV{TORCH_HOME})
endif ()
if (CMAKE_PREFIX_PATH)
    list(REMOVE_DUPLICATES CMAKE_PREFIX_PATH)
endif ()
message(STATUS "Looking for packages in : ${CMAKE_PREFIX_PATH}")
# ---

# check the compiler version
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    if (NOT CMAKE_CXX_COMPILER_VERSION) #Compiler version is not set on Ubuntu 12.02 (gcc 4.6)
        execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE CMAKE_CXX_COMPILER_VERSION)
    endif ()
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 6.0)
        message("CXX COMPILER VERSION: ${CMAKE_CXX_COMPILER_ID} : ${CMAKE_CXX_COMPILER_VERSION}")
        message(FATAL_ERROR "Gadgetron requires GCC version >= 6.0")
    endif ()
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 3.5)
        message(FATAL_ERROR "Gadgetron requires Clang version >= 3.5")
    endif ()
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 18.0)
        message(FATAL_ERROR "Gadgetron requires MSVC 2013 or above ")
    endif ()
else ()
    message(WARNING "Unsupported compiler!")
endif ()

set(CMAKE_INSTALL_RPATH ".:${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)


list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
set(GADGETRON_INSTALL_INCLUDE_PATH include/gadgetron)

# build options for OpenMP support
find_package(OpenMP)
option(USE_OPENMP "Use OpenMP" On)
if (OPENMP_FOUND)
    if (USE_OPENMP)
        message("OpenMP multithreading enabled")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        add_definitions(-DUSE_OMP)
    else ()
        message("OpenMP multithreading is supported, but disabled")
    endif ()
else ()
    message("OpenMP multithreading not supported")
endif ()

# for non-container efforts..
link_directories("${CMAKE_INSTALL_PREFIX}/lib")
include_directories("${CMAKE_INSTALL_PREFIX}/include/gadgetron/")

find_package(Gadgetron REQUIRED)
add_definitions(-DARMA_DONT_USE_WRAPPER -DARMA_USE_CXX11 -DARMA_64BIT_WORD)
#find_package(CUDA_advanced)
#find_package(Boost REQUIRED)
#find_package(Python3_compat REQUIRED)
find_package(Armadillo REQUIRED )

if (USE_MKL)
    if (LINUX)
        list(APPEND CMAKE_PREFIX_PATH "/opt/intel/oneapi/mkl/latest")
    endif()
    find_package(MKL REQUIRED)
    link_directories(GTBLAS BEFORE INTERFACE "${MKL_ROOT}/lib/intel64/")
    add_library(GTBLAS INTERFACE)
    set_property(TARGET GTBLAS PROPERTY INTERFACE_LINK_LIBRARIES ${MKL_LIBRARIES})
    message("MKL LIB DIR ${MKL_ROOT}/lib/intel64")

    add_definitions(-DUSE_MKL -DARMA_BLAS_LONG_LONG -DARMA_USE_BLAS -DARMA_USE_LAPACK)
else ()
    find_package(CBLAS REQUIRED)
    find_package(LAPACK REQUIRED)
    find_package(LAPACKE REQUIRED)
    add_library(GTBLAS INTERFACE)
    set_property(TARGET GTBLAS PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CBLAS_INCLUDE_DIR} ${LAPACK_INCLUDE_DIR} ${LAPACKE_INCLUDE_DIR})
    set_property(TARGET GTBLAS PROPERTY INTERFACE_LINK_LIBRARIES ${CBLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${LAPACKE_LIBRARIES})

    find_package(PkgConfig)
    pkg_check_modules(PC_lapacke lapacke)

    string(COMPARE EQUAL ${BLA_VENDOR} "OpenBLAS" USING_OPENBLAS)
    if (USING_OPENBLAS)
        message("OpenBLAS detected; setting max number of OpenBLAS pthreads to one.")
        add_definitions(-DFORCE_LIMIT_OPENBLAS_NUM_THREADS)
    endif()
endif ()
#find_package(CBLAS REQUIRED)
#find_package(LAPACK REQUIRED)
#find_package(cpr REQUIRED)
#find_package(ITK REQUIRED)
#include(${ITK_USE_FILE})


#set(Boost_USE_STATIC_LIBS OFF)
#set(Boost_USE_MULTITHREADED ON)
#set(Boost_USE_STATIC_RUNTIME OFF)
# necessary for Windows and RHEL <=6 systems

if(WIN32)
    add_definitions( -DHAVE_SNPRINTF) #Python decides to overwrite snprintf if this is not defined, because ofcourse it does.
endif()
# We actually only use system and thread explicitly, but they require linking in date_time and chrono
if (WIN32)
    find_package(Boost 1.71.0 COMPONENTS coroutine system date_time chrono program_options filesystem timer exception REQUIRED )
else()
    find_package(Boost 1.71.0 COMPONENTS coroutine system date_time program_options filesystem timer REQUIRED )
endif()
add_definitions("-DBOOST_BIND_GLOBAL_PLACEHOLDERS") # To suppress messages in later versions of boost, in version 1.76.0 there are internal conflicts in boost
add_definitions("-DBOOST_ALLOW_DEPRECATED_HEADERS") # To suppress messages in later versions of boost, in version 1.76.0 there are internal conflicts in boost
if (NOT DEFINED Boost_VERSION_STRING)
    set(Boost_VERSION_STRING "${Boost_MAJOR_VERSION}.${Boost_MINOR_VERSION}.${Boost_SUBMINOR_VERSION}")
endif()

if (BUILD_PYTHON_SUPPORT)
    find_package(Python3 COMPONENTS Development NumPy REQUIRED)
    find_package(Boost COMPONENTS python3${Python3_VERSION_MINOR} REQUIRED)

    message("Found numpy : ${Python3_NumPy_INCLUDE_DIRS} ${Python3_NumPy_LIBRARIES} ${Python3_INCLUDE_DIRS} ${Python3_LIBRARIES}")

    if (Python3_VERSION_MINOR LESS 7)
        message(FATAL_ERROR "Gadgetron Python support requires Python 3.7 or newer. Found: ${Python3_VERSION}")
    endif()

    message("Boost Version: ${Boost_VERSION_STRING} (with Python ${Python3_VERSION})")
    message("Secret: ${Boost_PYTHON3_TARGET}")

    add_library(python INTERFACE)
    set_property(TARGET python PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
    target_link_libraries(python INTERFACE Python3::Python Python3::NumPy Boost::python3${Python3_VERSION_MINOR})

    install(TARGETS python EXPORT gadgetron-export)
endif()

find_package(ZFP)


if (ZFP_FOUND)
   add_definitions(-DGADGETRON_COMPRESSION_ZFP)
endif()

if (ZFP_FOUND)
   include_directories(${ZFP_INCLUDE_DIR})
endif()

option(USE_CUDA "Build with CUDA support" On)

if (USE_CUDA)
    find_package(CUDA_advanced)
endif()

if (CUDA_FOUND AND USE_CUDA)

    if (NOT CUDA_cublas_device_LIBRARY)
        list(REMOVE_ITEM CUDA_CUBLAS_LIBRARIES CUDA_cublas_device_LIBRARY-NOTFOUND)
    endif()

    add_definitions(-DUSE_CUDA)
    set( GADGETRON_CUDA_FOUND_BOOL 1 )
    include_directories( ${CUDA_INCLUDE_DIRS} )
    set(CUDA_PROPAGATE_HOST_FLAGS ON)
    set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --std=c++17")
    set(CMAKE_CUDA_STANDARD 17)


    # Compile kernels for compute models 1.0 and 2.0 as default for Cuda 4.1
    # Support compute model 3.0 from Cuda 4.2 and up
    # Support compute model 3.5 from Cuda 5 and up

    option(GADGETRON_CUDA_ALL_COMPUTE_MODEL "Build CUDA components for all computing models" OFF)
    if (GADGETRON_CUDA_ALL_COMPUTE_MODEL)
        message("Compiling CUDA components to support all compute model 3.5, 5.0, 5.2, 6.0, 6.1, 7.0, 8.0, 8.6, 9.0")
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_35,code=sm_35")
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_50,code=sm_50")
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_52,code=sm_52")
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_60,code=sm_60")
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_61,code=sm_61")
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_70,code=sm_70")
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_75,code=sm_75")
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_80,code=sm_80")
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_86,code=sm_86")
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_90,code=sm_90")
    endif ()

    message("Compiling with ${CUDA_NVCC_FLAGS}")
else ()
    message("CUDA not found. CUDA components will not be compiled.")
    set( GADGETRON_CUDA_FOUND_BOOL 0 )
    set( CUDA_NVCC_FLAGS "NA")
endif ()
set(CMAKE_CXX_STANDARD 17)

set( gadgetron_nhlbi_gt_toolbox_header_files
    utils/FeedbackSNR/AcquisitionAccumulateTriggerGadget_LIT.h
    Feedback/readers/FeedbackReader.h
    Feedback/writers/FeedbackWriter.h
    Feedback/FeedbackData.h
    Feedback/MessageID.h
)
   
set( gadgetron_nhlbi_gt_toolbox_src_files
    utils/FeedbackSNR/ImageAndFeedback.cpp	
    utils/FeedbackSNR/AcquisitionAccumulateTriggerGadget_LIT.cpp
    utils/FeedbackSNR/FeedbackTestGadget.cpp
    utils/FeedbackSNR/AverageReconGadget.cpp
    utils/Flow/AutoScaleFlow.cpp
    Feedback/readers/FeedbackReader.cpp
    Feedback/writers/FeedbackWriter.cpp
    Feedback/gadgetron_ismrmrd_client_feedback/gadgetron_ismrmrd_client_feedback.cpp
)

set( gadgetron_nhlbi_gt_toolbox_config_files
    config/mrd_dump.xml
 )
 

add_library(nhlbi_gt_toolbox SHARED  
${gadgetron_nhlbi_gt_toolbox_header_files} 
${gadgetron_nhlbi_gt_toolbox_src_files}
${gadgetron_nhlbi_gt_toolbox_config_files}
)
target_include_directories(nhlbi_gt_toolbox
 PUBLIC    
 $<INSTALL_INTERFACE:include>
 ${CMAKE_PREFIX_PATH}/include/
 ${CMAKE_PREFIX_PATH}/include/gadgetron/
 ${CMAKE_PREFIX_PATH}/include/gadgetron/io/ 
 $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/>
 $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/>
 $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/utils/>
 /opt/code/cuda-samples/Common
)

target_link_libraries(nhlbi_gt_toolbox gadgetron_core 
                                       gadgetron_toolbox_cpucore_math 
                                       gadgetron_toolbox_spiral 
                                       gadgetron_toolbox_cpunfft 
                                       gadgetron_toolbox_cpusdc 
                                       gadgetron_toolbox_cpureg
                                       gadgetron_mricore
									   gadgetron_toolbox_cpufft
									   gadgetron_toolbox_mri_core
									   gadgetron_toolbox_log
									   gadgetron_toolbox_cpuoperator
									   gadgetron_toolbox_image_analyze_io
									   gadgetron_core_parallel )


	target_include_directories(nhlbi_gt_toolbox
    PUBLIC    
	$<INSTALL_INTERFACE:include>
	${CMAKE_PREFIX_PATH}/include/
	${CMAKE_PREFIX_PATH}/include/gadgetron/
	${CMAKE_PREFIX_PATH}/include/gadgetron/io/ 
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/utils/>
	/opt/code/cuda-samples/Common
)
include_directories(
	${CMAKE_SOURCE_DIR}/utils/motionCorrection/
)

install(FILES ${gadgetron_nhlbi_gt_toolbox_header_files} ${gadgetron_nhlbi_gt_toolbox_src_files}  
DESTINATION ${GADGETRON_INSTALL_INCLUDE_PATH} COMPONENT main)

add_subdirectory(utils)



target_link_libraries(nhlbi_gt_toolbox
	nhlbi_gt_utils)
 


	cuda_add_library(nhlbi_toolbox_gpu SHARED
	utils/gpu/gpuregistration/gpuRegistration.cu
	utils/gpu/gpuregistration/gpuRegistration.cuh
	${gadgetron_nhlbi_gpu_recon_header_files} 
	${gadgetron_nhlbi_gpu_recon_src_files}
)
set_target_properties(nhlbi_toolbox_gpu PROPERTIES VERSION ${GADGETRON_VERSION_STRING} SOVERSION ${GADGETRON_SOVERSION})

#target_link_libraries(nhlbi_toolbox_gpu gadgetron_core gadgetron_toolbox_cpucore_math gadgetron_toolbox_spiral gadgetron_toolbox_cpunfft)

target_link_libraries(nhlbi_toolbox_gpu
gadgetron_toolbox_hostutils 
gadgetron_toolbox_mri_core
gadgetron_toolbox_cpusdc
gadgetron_toolbox_gpu
gadgetron_mri_noncartesian
gadgetron_toolbox_spiral 
gadgetron_toolbox_cpunfft
gadgetron_toolbox_log
gadgetron_mricore
gadgetron_toolbox_cpufft
gadgetron_toolbox_mri_core
gadgetron_toolbox_log
gadgetron_toolbox_cpuoperator
gadgetron_toolbox_image_analyze_io
gadgetron_core_parallel
gadgetron_toolbox_cpusdc
gadgetron_toolbox_gpu
gadgetron_core
gadgetron_toolbox_cpucore_math
gadgetron_toolbox_spiral
gadgetron_toolbox_cpunfft
${CUDA_LIBRARIES}
${CUDA_CUFFT_LIBRARIES} 
${CUDA_CUBLAS_LIBRARIES} 
${CUDA_CUSPARSE_LIBRARIES}
${MKL_LIBRARIES}
)

target_include_directories(nhlbi_toolbox_gpu
	PUBLIC    
	$<INSTALL_INTERFACE:include>
$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/>
$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/>
$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/utils/>
$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/utils/gpu/>
${CMAKE_PREFIX_PATH}/lib/
${CMAKE_PREFIX_PATH}/include/
$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/utils/gpu/gpuregistration/>
$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/utils/gpu/gpuregistration/internal/>
$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/>
$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}
${CMAKE_PREFIX_PATH}/include/
${CMAKE_PREFIX_PATH}/include/gadgetron/
${CMAKE_PREFIX_PATH}/include/gadgetron/io/ 
/opt/code/cuda-samples/Common
${CUDA_INCLUDE_DIRS}
)

install(TARGETS nhlbi_toolbox_gpu
LIBRARY DESTINATION lib
ARCHIVE DESTINATION lib
RUNTIME DESTINATION bin
COMPONENT main
)

install(FILES
utils/gpu/cuda_utils.h
utils/gpu/gpuregistration/cubicTex3D.cu
utils/gpu/gpuregistration/cubicPrefilter3D.cu
utils/gpu/gpuregistration/gpuRegistration.cu
utils/gpu/gpuregistration/gpuRegistration.cuh
${gadgetron_nhlbi_gpu_recon_header_files}
${gadgetron_nhlbi_gpu_recon_src_files}
DESTINATION ${GADGETRON_INSTALL_INCLUDE_PATH}
COMPONENT main)


target_link_libraries(nhlbi_gt_toolbox
nhlbi_toolbox_gpu -lcurand)

install(FILES ${gadgetron_nhlbi_gt_toolbox_config_files} DESTINATION ${GADGETRON_INSTALL_CONFIG_PATH} COMPONENT main)
install(TARGETS nhlbi_gt_toolbox
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
	RUNTIME DESTINATION bin
	COMPONENT main
)

add_executable(gadgetron_ismrmrd_client_feedback Feedback/gadgetron_ismrmrd_client_feedback/gadgetron_ismrmrd_client_feedback.cpp)
include_directories(${HDF5_INCLUDE_DIRS})
target_link_libraries(gadgetron_ismrmrd_client_feedback ISMRMRD::ISMRMRD Boost::program_options gadgetron_mricore nhlbi_gt_toolbox)

if (ZFP_FOUND)
   target_link_libraries(gadgetron_ismrmrd_client_feedback ${ZFP_LIBRARIES})
endif ()

install(TARGETS gadgetron_ismrmrd_client_feedback DESTINATION bin COMPONENT main)