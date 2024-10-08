cmake_minimum_required(VERSION 3.8)

# Specify the CUDA compiler and architectures
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.5/bin/nvcc")
set(CMAKE_CUDA_ARCHITECTURES "75")

project(fdmtk LANGUAGES CXX CUDA)

# Set the C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Ensure CUDA language is enabled
include(CheckLanguage)
check_language(CUDA)
enable_language(CUDA)

# Set CUDA include directories
include_directories(/usr/local/cuda-12.5/targets/x86_64-linux/include/)
include_directories(/usr/local/cuda-12.5/targets/x86_64-linux/include/cub)
include_directories(${CMAKE_SOURCE_DIR})

# Explicitly set the HDF5 include and library directories
set(HDF5_INCLUDE_DIRS "/usr/include/hdf5/serial")
set(HDF5_LIBRARIES "/usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5.so")

# Include HDF5 directories
include_directories(${HDF5_INCLUDE_DIRS})

set(SRC_FILES
    main.cpp    
    ChunkB.cpp
    Chunk_fly_gpu.cu
    Chunk_gpu.cu    
    FdmtB.cpp
    FdmtCpu.cpp
    FdmtGpu.cu    
    Candidate.cu
    Clusterization.cu
    Segmented_sort.cu
    select_delegates.cu
    OutChunkHeader.cpp
    SessionB.cpp
    Session_guppi.cpp  
    Session_lofar.cpp  
    Session_lofar_gpu.cu
    Statistical_preprocessing.cu    
    Session_guppi_gpu.cu
    TelescopeHeader.cpp
)

# Add executable target
add_executable(CUFDMT ${SRC_FILES})

# Link CUDA libraries
target_link_libraries(CUFDMT cudadevrt cudart_static rt pthread dl)

# Link HDF5 libraries explicitly
target_link_libraries(CUFDMT ${HDF5_LIBRARIES})

# Link cuFFT library
target_link_libraries(CUFDMT cufft)

