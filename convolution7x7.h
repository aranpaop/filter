#pragma once

#include "../common.h"

class Convolution7x7;

class ConvSample final : public Sample {
public:
    ConvSample(cl_int, cl_int);
    //~ConvSample();
    ConvSample(const ConvSample&) = delete;
    ConvSample(ConvSample&&) = delete;
    ConvSample& operator=(const ConvSample&) = delete;
    ConvSample& operator=(ConvSample&&) = delete;

    void Run() override;

    Convolution7x7* m_conv7x7;
};

class Convolution7x7 final : public Algorithm {
public:
    Convolution7x7(std::vector<std::vector<size_t>>&, ConvSample*);
    ~Convolution7x7();
    Convolution7x7(const Convolution7x7&) = delete;
    Convolution7x7(Convolution7x7&&) = delete;
    Convolution7x7& operator=(const Convolution7x7&) = delete;
    Convolution7x7& operator=(Convolution7x7&&) = delete;

    void CreateKernel(const std::string&) override;
    void RunWithLsize(std::vector<size_t>&, bool) override;
private:
    void PrepareData(WisePtr<cl_float>&, WisePtr<cl_float>&, WisePtr<cl_int>&);
    void ValidateResult(WisePtr<cl_float>&, WisePtr<cl_float>&, WisePtr<cl_float>&, WisePtr<cl_int>&);
    ConvSample* m_sample;
    cl_mem m_src;
    cl_mem m_dst;
    cl_mem m_filter;
    cl_mem m_map;
};
