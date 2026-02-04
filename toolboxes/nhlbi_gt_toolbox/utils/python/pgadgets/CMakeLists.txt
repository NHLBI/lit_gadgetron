include_directories(${CMAKE_PREFIX_PATH}/include/ ${ISMRMRD_INCLUDE_DIR} )

# add_executable(gadgetron_test_python test_python.cpp)
# target_link_libraries(gadgetron_test_python gadgetron_toolbox_python Boost::filesystem)

add_executable(python_demo_nhlbi demo.cpp)
target_link_libraries(python_demo_nhlbi python 
gadgetron_core
gadgetron_toolbox_cpucore_math
gadgetron_toolbox_python 
gadgetron_toolbox_log 
Boost::filesystem
ismrmrd
)

target_include_directories(python_demo_nhlbi
    PUBLIC
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS} 
    ${Python3_NumPy_LIBRARIES} 
    ${Python3_INCLUDE_DIRS} 
    ${Python3_LIBRARIES}
    ${ISMRMRD_INCLUDE_DIR}
    ${CMAKE_PREFIX_PATH}/include/
    ${CMAKE_PREFIX_PATH}/include/
    ${CMAKE_PREFIX_PATH}/include/io/

)

# install(TARGETS gadgetron_test_python DESTINATION bin COMPONENT main)
install(TARGETS python_demo_nhlbi DESTINATION bin COMPONENT main)
