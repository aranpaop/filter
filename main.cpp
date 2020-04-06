#include <iostream>
#include <string>
#include <CL/opencl.h>

#include "convolution7x7/convolution7x7.h"

void TestConvolution7x7(cl_int width, cl_int height)
{
    cl_platform_id platform = GetPlatform("NVIDIA CUDA");

    ConvSample sample{width, height};
    sample.SetupEnv(platform, "convolution7x7/convolution7x7.cl");

    std::vector<std::vector<size_t>> testLsizes{
        {1, 1},
        {2, 1},
        {2, 2},
        {4, 2},
        {4, 4},
        {8, 4},
        {8, 8},
        {16, 4},
        {16, 8}
    };
    Convolution7x7 conv7x7{ testLsizes, &sample };
    sample.m_conv7x7 = &conv7x7;

    sample.Run();
}

int main(int argc, char** argv)
{
    cl_int width{2560};
    cl_int height{1440};
    bool testConvolution7x7{ false };
    bool testUpsampling4x4{ false };
    for (int i = 1; i < argc; ++i) {
        std::string arg{ argv[i] };
        if (arg == "convolution7x7") {
            testConvolution7x7 = true;
        }
        else if (arg == "Upsampling4x4") {
            testUpsampling4x4 = true;
        }
        else if (arg[0] == 'w') {
            try {
                width = std::stoi(arg.substr(1));
            }
            catch (...) {
                std::cout << "Unknown input parameter, exiting..." << std::endl;
                exit(EXIT_FAILURE);
            }
        }
        else if (arg[0] == 'h') {
            try {
                height = std::stoi(arg.substr(1));
            }
            catch (...) {
                std::cout << "Unknown input parameter, exiting..." << std::endl;
                exit(EXIT_FAILURE);
            }
        }
        else {
            std::cout << "Unknown input parameter, exiting..." << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    if (testConvolution7x7) {
        std::cout << "Start testing convolution7x7 with width: " << width << ", height: " << height << std::endl;
        TestConvolution7x7(width, height);
        std::cout << "End testing convolution7x7" << std::endl;
    }

    return 0;
}