#include "common.h"

Sample::Sample(cl_int width, cl_int height) : m_width{ width }, m_height{ height },
    m_device{ nullptr }, m_context{ nullptr }, m_queue{ nullptr }, m_program{ nullptr } {} 

Sample::~Sample()
{
    cl_int err;
    if (m_device != nullptr) {
        err = clReleaseDevice(m_device);
        CheckClErr(err, "clReleaseDevice");
    }
    if (m_context != nullptr) {
        err = clReleaseContext(m_context);
        CheckClErr(err, "clReleaseContext");
    }
    if (m_queue != nullptr) {
        err = clReleaseCommandQueue(m_queue);
        CheckClErr(err, "clReleaseCommandQueue");
    }
    if (m_program != nullptr) {
        err = clReleaseProgram(m_program);
        CheckClErr(err, "clReleaseProgram");
    }
}

void Sample::SetupEnv(cl_platform_id platform, const std::string& sourceFile)
{
    cl_int err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &m_device, nullptr);
    CheckClErr(err, "clGetDeviceIDs");

    m_context = clCreateContext(nullptr, 1, &m_device, nullptr, nullptr, &err);
    CheckClErr(err, "clGetDeviceIDs");

    m_queue = clCreateCommandQueue(m_context, m_device, CL_QUEUE_PROFILING_ENABLE, &err);
    CheckClErr(err, "clCreateCommandQueue");

    std::string source;
    ReadSourceFile(sourceFile, source);
    std::cout << source << std::endl;
    const char* string = source.c_str();
    m_program = clCreateProgramWithSource(m_context, 1, reinterpret_cast<const char**>(&string), nullptr, &err);
    CheckClErr(err, "clCreateProgramWithBinary");
    err = clBuildProgram(m_program, 1, &m_device, nullptr, nullptr, nullptr);
    CheckClErr(err, "clBuildProgram");
}

Algorithm::Algorithm(std::vector<std::vector<size_t>>& testLsizes) : m_kernel{ nullptr }, m_bestLsize{ {8, 4} },
    m_testLsizes{ std::move(testLsizes) }, m_minExeTime{ 0xffffffffffffffff } {}

Algorithm::~Algorithm()
{
    if (m_kernel != nullptr) {
        cl_int err = clReleaseKernel(m_kernel);
        CheckClErr(err, "clReleaseKernel");
    }
}

void Algorithm::Tunning()
{
    std::for_each(m_testLsizes.begin(), m_testLsizes.end(), [&](std::vector<size_t>& testLsize) {
        RunWithLsize(testLsize, true);
    });
}
