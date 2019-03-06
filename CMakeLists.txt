cmake_minimum_required(VERSION 3.1)
project(transmitChannelizer)

set(PROJECT_VERSION 1)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

message("Looking for Boost")
find_package(Boost REQUIRED)
find_package(OpenCL REQUIRED)
find_package(Volk REQUIRED)

message(STATUS "Volk includes: ${VOLK_INCLUDE_DIRS}")

link_directories(${CMAKE_INSTALL_PREFIX}/bin/lib)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/bin/lib")


### Volk Test
add_executable(volk_test volk_test.cpp)
target_include_directories(volk_test PRIVATE ${VOLK_INCLUDE_DIRS})
## Set a specific volk library to use
add_library( volk UNKNOWN IMPORTED)
set_property(TARGET volk PROPERTY IMPORTED_LOCATION "/usr/local/lib/libvolk.so.1.4")

target_link_libraries(volk_test PUBLIC pthread volk fftw3_threads fftw3 m)
install(TARGETS volk_test RUNTIME DESTINATION bin)



### Open CL Test
add_executable(opencl_test opencl_test.cpp)
target_link_libraries(opencl_test PUBLIC pthread OpenCL)

install(TARGETS opencl_test RUNTIME DESTINATION bin)

file(GLOB KERNELS "*.cl")
install(FILES ${KERNELS} DESTINATION bin/kernels)
