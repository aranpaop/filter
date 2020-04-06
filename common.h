#pragma once

#include <vector>
#include <string>
#include <CL/opencl.h>

#include "utils.h"

class Sample {
public:
    Sample(cl_int, cl_int);
    ~Sample();
    Sample(const Sample&) = delete;
    Sample(Sample&&) = delete;
    Sample& operator=(const Sample&) = delete;
    Sample& operator=(Sample&&) = delete;

    void SetupEnv(cl_platform_id, const std::string&);
    virtual void Run() = 0;

    cl_int m_width;
    cl_int m_height;
    cl_device_id m_device;
    cl_context m_context;
    cl_command_queue m_queue;
    cl_program m_program;
};

class Algorithm {
public:
    Algorithm(std::vector<std::vector<size_t>>&);
    ~Algorithm();
    Algorithm(const Algorithm&) = delete;
    Algorithm(Algorithm&&) = delete;
    Algorithm& operator=(const Algorithm&) = delete;
    Algorithm& operator=(Algorithm&&) = delete;

    virtual void CreateKernel(const std::string&) = 0;
    void Tunning();
    virtual void RunWithLsize(std::vector<size_t>&, bool) = 0;

    cl_kernel m_kernel;
    std::vector<size_t> m_bestLsize;
    std::vector<std::vector<size_t>> m_testLsizes;
    cl_ulong m_minExeTime;
};
