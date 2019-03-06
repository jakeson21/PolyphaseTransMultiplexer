#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
//#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#endif

#define MAX_SOURCE_SIZE (0x100000)

int main(int argc, char **argv) {
    std::cout << "Hello, world!" << std::endl;
    
    // Create the two input vectors
    int i;
    const int LIST_SIZE = 1024;
    std::vector<int> A(LIST_SIZE, 0);
    std::vector<int> B(LIST_SIZE, 0);
    for(i = 0; i < LIST_SIZE; i++) {
        A[i] = i;
        B[i] = LIST_SIZE - i;
    }
 
    // Load the kernel source code into the array source_str
    FILE *fp;
    char *source_str;
    size_t source_size;
 
    fp = fopen("kernels/vector_add_kernel.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose( fp );
 
    
    // Get platform and device information
    //get all platforms (drivers)
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if(all_platforms.size()==0){
        std::cout<<" No platforms found. Check OpenCL installation!\n";
        exit(1);
    }
    std::cout << "Found platforms:" << std::endl;
    for (auto& p : all_platforms)
    {
        std::cout << p.getInfo<CL_PLATFORM_NAME>() << std::endl;
    }    
    cl::Platform default_platform=all_platforms[0];
    std::cout << "\nUsing platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<"\n";
    
    //get default device of the default platform
    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if(all_devices.size()==0){
        std::cout<<" No devices found. Check OpenCL installation!\n";
        exit(1);
    }
    std::cout << "Found devices:" << std::endl;
    for (auto& d : all_devices)
    {
        std::cout << d.getInfo<CL_DEVICE_NAME>() << std::endl;
    }
    cl::Device default_device=all_devices[0];
    std::cout<< "\nUsing device: "<<default_device.getInfo<CL_DEVICE_NAME>()<<"\n";
        
    cl::Context context({default_device});
 
    cl::Program::Sources sources;
    std::string kernel(source_str);
    
    std::cout << "Using kernel:\n" << source_str << std::endl;
    
    sources.push_back({kernel.c_str(), kernel.size()});
    
    cl::Program program(context, sources);
    if(program.build({default_device}) != CL_SUCCESS){
        std::cout<<" Error building: "<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device)<<"\n";
        exit(1);
    }

    std::cout << "Creating buffers on the device" << std::endl;
    // create buffers on the device
    cl_int err = 0;
    cl::Buffer buffer_A(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int)*LIST_SIZE, &err);
    std::cout << "buffer_A: " << err << std::endl;
    cl::Buffer buffer_B(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int)*LIST_SIZE, &err);
    std::cout << "buffer_B: " << err << std::endl;
    cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, sizeof(int)*LIST_SIZE, &err);
    std::cout << "buffer_C: " << err << std::endl;
 
    std::cout << "Creating CommandQueue" << std::endl;
    //create queue to which we will push commands for the device.
    cl::CommandQueue queue(context, default_device);

    std::cout << "Writing arrays A and B to the device" << std::endl;
    //write arrays A and B to the device
    queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(int)*LIST_SIZE, A.data());
    queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, sizeof(int)*LIST_SIZE, B.data());
 
    //run the kernel
    cl::Kernel simple_add(program, "VectorAdd", &err);
    std::cout << "VectorAdd: " << err << std::endl;
    std::cout << "setArg" << std::endl;
    err = simple_add.setArg(0, buffer_A);
    std::cout << "setArg 0: " << err << std::endl;
    err = simple_add.setArg(1, buffer_B);
    std::cout << "setArg 1: " << err << std::endl;
    err = simple_add.setArg(2, buffer_C);
    std::cout << "setArg 2: " << err << std::endl;
    err = simple_add.setArg(3, static_cast<cl_ulong>(LIST_SIZE));
    std::cout << "setArg 3: " << err << std::endl;
    
    std::cout << "enqueueNDRangeKernel" << std::endl;
    std::cout << "1" << std::endl;
    for (int n=0; n<2000000; n++)
        queue.enqueueNDRangeKernel(simple_add, cl::NullRange, LIST_SIZE, 1);
    std::cout << "4" << std::endl;
    for (int n=0; n<2000000; n++)
        queue.enqueueNDRangeKernel(simple_add, cl::NullRange, LIST_SIZE, 4);
    std::cout << "16" << std::endl;
    for (int n=0; n<2000000; n++)
        queue.enqueueNDRangeKernel(simple_add, cl::NullRange, LIST_SIZE, 16);
    std::cout << "32" << std::endl;
    for (int n=0; n<2000000; n++)
        queue.enqueueNDRangeKernel(simple_add, cl::NullRange, LIST_SIZE, 32);
        

    std::vector<int> C(LIST_SIZE, 0);
    //read result C from the device to array C
    queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(int)*C.size(), C.data());
    
    
    std::cout<<" result: \n";
    for(int i = 0; i < C.size(); i++){
        std::cout << A[i] << " + " << B[i] << " = " << C[i] << std::endl;
    }
    
    return 0;    
}
