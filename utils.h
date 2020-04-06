#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <initializer_list>
#include <algorithm>
#include <CL/opencl.h>

inline void CheckClErr(cl_int err, const std::string& func)
{
    if (err != CL_SUCCESS) {
        std::cout << func << " filed, err code " << err << std::endl;
        exit(EXIT_FAILURE);
    }
    std::cout << func << " success" << std::endl;
}

inline cl_platform_id GetPlatform(const std::string& platformName)
{
    cl_uint numPlatform;
    cl_int err = clGetPlatformIDs(0, nullptr, &numPlatform);
    CheckClErr(err, "clGetPlatformIDs");

    auto platforms(std::make_unique<cl_platform_id[]>(numPlatform));
    err = clGetPlatformIDs(numPlatform, platforms.get(), nullptr);
    CheckClErr(err, "clGetPlatformIDs");

    cl_platform_id platform{ nullptr };
    for (cl_uint i = 0; i < numPlatform; ++i) {
        auto name(std::make_unique<char[]>(47));
        cl_int err = clGetPlatformInfo(platforms.get()[i], CL_PLATFORM_NAME, 47, name.get(), nullptr);
        std::string sname = name.get();
        if (sname == platformName) {
            std::cout << "Platform name: " << platformName << std::endl;
            platform = platforms.get()[i];
            break;
        }
    }

    if (platform == nullptr) {
        std::cout << "Failed to get correct platform." << std::endl;
        exit(EXIT_FAILURE);
    }

    return platform;
}

inline void ReadSourceFile(const std::string& sourceFile, std::string& source)
{
    std::ifstream fs(sourceFile);
    if (!fs.is_open()) {
        std::cout << "Failed to open source file." << std::endl;
        exit(EXIT_FAILURE);
    }
    std::ostringstream ss;
    ss << fs.rdbuf();
    fs.close();
    source = ss.str();
}

inline void ReleaseClMemObj(std::initializer_list<cl_mem>& objs)
{
    std::for_each(objs.begin(), objs.end(), [](cl_mem obj) {
        if (obj != nullptr) {
            cl_int err = clReleaseMemObject(obj);
            CheckClErr(err, "clReleaseMemObject");
        }
    });
}

template<typename T>
inline void FillRandomData(T* ptr, size_t size, T bottom, T top)
{
    srand(0);
    for (size_t i = 0; i < size; ++i) {
        ptr[i] = bottom + static_cast<T>((static_cast<cl_double>(top) - static_cast<cl_double>(bottom)) *
            static_cast<cl_double>(rand()) / static_cast<cl_double>(RAND_MAX));
        if (ptr[i] >= top) { ptr[i] = (bottom + top) / 2; }
    }
}

template<typename T, typename ... Ts>
void SetKernelArgs(cl_kernel kernel, cl_uint index, T arg, Ts ... args)
{
    cl_int err = clSetKernelArg(kernel, index, sizeof(T), &arg);
    CheckClErr(err, "clSetKernelArg");
    ++index;
    SetKernelArgs(kernel, index, args ...);
}

template<typename T>
void SetKernelArgs(cl_kernel kernel, cl_uint index, T arg)
{
    cl_int err = clSetKernelArg(kernel, index, sizeof(T), &arg);
    CheckClErr(err, "clSetKernelArg");
}

inline cl_ulong GetProfilingInfo(cl_event event)
{
    cl_ulong queuedTime, submitTime, startTime, endTime;
    cl_int err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &queuedTime, nullptr);
    err |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong), &submitTime, nullptr);
    err |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &startTime, nullptr);
    err |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endTime, nullptr);
    CheckClErr(err, "clGetEventProfilingInfo");
    std::cout << "Time between queued and submit: " << submitTime - queuedTime << std::endl;
    std::cout << "Time between submit and start: " << startTime - submitTime << std::endl;
    std::cout << "Time between start and end: " << endTime - startTime << std::endl;

    return endTime - startTime;
}

inline cl_float Ulp(cl_float got, cl_float expected)
{
    //cl_float next = std::nextafterf(got, expected);
    //return abs(got - next);
    return 0.01f;
}

template<typename T>
class WisePtr {
public:
    explicit WisePtr(size_t size) : m_size{ size }, m_ptr{ std::make_unique<T[]>(size) } {}
    WisePtr(const WisePtr&) = delete;
    WisePtr(WisePtr&&) = delete;
    WisePtr& operator=(const WisePtr&) = delete;
    WisePtr& operator=(WisePtr&&) = delete;

    T* get() { return m_ptr.get(); }
    size_t size() { return m_size; }

    T& operator[](size_t index)
    {
        if (index >= m_size) {
            std::cout << "Buffer index " << index << " out of range " << m_size << std::endl;
            exit(EXIT_FAILURE);
        }
        return m_ptr.get()[index];
    }
private:
    std::unique_ptr<T[]> m_ptr;
    size_t m_size;
};
