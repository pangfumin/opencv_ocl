 
cmake_minimum_required(VERSION 2.8)
project(ocl_module) 



find_package(OpenCV 3.0 QUIET)
# if(NOT OpenCV_FOUND)
#    find_package(OpenCV 2.4.3 QUIET)
#    if(NOT OpenCV_FOUND)
#       message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
#    endif()
# endif()

find_package(OpenCL 1.2 QUIET)
if(NOT OpenCL_FOUND)
message(STATUS "OpenCL samples are skipped: OpenCL SDK is required")
return()
endif()

include_directories(include 
src
${OpenCV_INCLUDE_DIR}
${OpenCL_INCLUDE_DIR})

add_library(${PROJECT_NAME}
    src/tvl1flow.cpp
    # src/svm.cpp
    # src/stereo_csbp.cpp
    # src/stereobp.cpp
    # src/stereobm.cpp
    # src/split_merge.cpp
    # src/sort_by_key.cpp
    # src/safe_call.hpp
    src/pyrup.cpp
    src/pyrlk.cpp
    src/pyrdown.cpp
    src/precomp.hpp
    src/optical_flow_farneback.cpp
    src/opencl
    src/mssegmentation.cpp
    src/moments.cpp
    src/mcwutil.cpp
    src/matrix_operations.cpp
    src/match_template.cpp
    # src/knearest.cpp
    # src/kmeans.cpp
    src/kalman.cpp
    src/interpolate_frames.cpp
    src/imgproc.cpp
    # src/hog.cpp
    # src/haar.cpp
    src/gftt.cpp
    # src/gemm.cpp
    src/filtering.cpp
    # src/fft.cpp
    src/error.cpp
    # src/columnsum.cpp
    src/color.cpp
    src/cl_runtime
    src/cl_programcache.hpp
    src/cl_programcache.cpp
    src/cl_operations.cpp
    src/cl_context.cpp
    # src/canny.cpp
    src/build_warps.cpp
    # src/brute_force_matcher.cpp
    # src/blend.cpp
    # src/bgfg_mog.cpp
    src/arithm.cpp
    src/opencl_kernels.cpp
    # src/cl_runtime/clamdblas_runtime.cpp  
    src/cl_runtime/cl_runtime.cpp              
    # src/cl_runtime/clamdfft_runtime.cpp   
    )

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${OpenCL_LIBRARY}
dl
)

add_executable(pyrlk_optical_flow samples/pyrlk_optical_flow.cpp)
target_link_libraries(pyrlk_optical_flow
    ${PROJECT_NAME}
${OpenCV_LIBS}
${OpenCL_LIBRARY}
dl
)

add_executable(test_optical_flow samples/test_optical_flow.cpp)
target_link_libraries(test_optical_flow
    ${PROJECT_NAME}
${OpenCV_LIBS}
${OpenCL_LIBRARY}
dl
)