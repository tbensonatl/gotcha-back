cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(sarbp LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(CUDAToolkit REQUIRED)
find_package(Boost COMPONENTS program_options REQUIRED)
find_package(GLUT REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)

set(DEFAULT_BUILD_TYPE "Release")
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' because none was specified")
    set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "Choose the build type." FORCE)
endif()

string(TOUPPER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE)
message(STATUS "Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -v -Xcompiler=-Wall -Xcompiler=-Wextra -Xcompiler=-Werror -Xptxas -lineinfo")
set(COMMON_CXX_FLAGS "-Wall -Wextra -Wswitch-enum -Werror -flto")

set(CMAKE_CXX_FLAGS_DEBUG "-g -O1 ${COMMON_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -g -O1")

set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g -O1 ${COMMON_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO} -g -O1")

set(CMAKE_CXX_FLAGS_RELEASE "-g -O3 -DNDEBUG ${COMMON_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -g -O3 -DNDEBUG")

add_library(kernels STATIC
    src/kernels.h
    src/kernels.cu)

set_target_properties(kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(kernels PROPERTIES CUDA_ARCHITECTURES "86")

set(CPP_SOURCES
    src/common.h
    src/common.cpp
    src/cpu_bp.h
    src/cpu_bp.cpp
    src/data_reader.h
    src/data_reader.cpp
    src/fft.h
    src/fft.cpp
    src/helpers.h
    src/helpers.cpp
    src/ser.h
    src/ser.cpp
    src/video_sar.h
    src/video_sar.cpp)

add_executable(sarbp src/main.cpp ${CPP_SOURCES})
set_target_properties(sarbp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(sarbp PROPERTIES CUDA_ARCHITECTURES "86")
target_include_directories(sarbp PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${Boost_INCLUDE_DIRS})
target_link_libraries(sarbp kernels ${Boost_LIBRARIES} CUDA::cufft ${GLUT_LIBRARIES} OpenGL::GL OpenGL::GLU GLEW::glew)
