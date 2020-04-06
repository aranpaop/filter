#include "convolution7x7.h"

ConvSample::ConvSample(cl_int width, cl_int height) : Sample{ width, height }, m_conv7x7{ nullptr } {}

void ConvSample::Run()
{
    m_conv7x7->CreateKernel("Convolution7x7");
    m_conv7x7->Tunning();
    m_conv7x7->RunWithLsize(m_conv7x7->m_bestLsize, false);
}

Convolution7x7::Convolution7x7(std::vector<std::vector<size_t>>& testLsizes, ConvSample* sample) : Algorithm{testLsizes},
    m_sample{ sample }, m_src{ nullptr }, m_dst{ nullptr }, m_filter{ nullptr }, m_map{nullptr} {}

Convolution7x7::~Convolution7x7()
{
    cl_int err;
    if (m_src != nullptr) {
        err = clReleaseMemObject(m_src);
        CheckClErr(err, "clReleaseMemObject");
    }
    if (m_dst != nullptr) {
        err = clReleaseMemObject(m_dst);
        CheckClErr(err, "clReleaseMemObject");
    }
    if (m_filter != nullptr) {
        err = clReleaseMemObject(m_filter);
        CheckClErr(err, "clReleaseMemObject");
    }
    if (m_map != nullptr) {
        err = clReleaseMemObject(m_map);
        CheckClErr(err, "clReleaseMemObject");
    }
}

void Convolution7x7::CreateKernel(const std::string& kernelName)
{
    cl_int err;
    m_kernel = clCreateKernel(m_sample->m_program, kernelName.c_str(), &err);
    CheckClErr(err, "clCreateKernel");
}

void Convolution7x7::PrepareData(WisePtr<cl_float>& src, WisePtr<cl_float>& filter, WisePtr<cl_int>& map)
{
    FillRandomData<cl_float>(src.get(), src.size(), -1.0f, 1.0f);
    FillRandomData<cl_float>(filter.get(), filter.size(), -1.0f, 1.0f);
    FillRandomData<cl_int>(map.get(), map.size(), 0, 10);

    cl_int err;
    m_src = clCreateBuffer(m_sample->m_context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                           src.size() * sizeof(cl_float), src.get(), &err);
    CheckClErr(err, "clCreateBuffer");
    m_dst = clCreateBuffer(m_sample->m_context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_WRITE_ONLY,
                           map.size() * sizeof(cl_float), nullptr, &err);
    CheckClErr(err, "clCreateBuffer");
    m_filter = clCreateBuffer(m_sample->m_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                              filter.size() * sizeof(cl_float), filter.get(), &err);
    CheckClErr(err, "clCreateBuffer");
    m_map = clCreateBuffer(m_sample->m_context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                           map.size() * sizeof(cl_int), map.get(), &err);
    CheckClErr(err, "clCreateBuffer");

    SetKernelArgs(m_kernel, 0, m_src, m_dst, m_filter, m_map, m_sample->m_width);
}

void::Convolution7x7::ValidateResult(WisePtr<cl_float>& src, WisePtr<cl_float>& dst, WisePtr<cl_float>& filter, WisePtr<cl_int>& map)
{
    for (cl_int row = 0; row < m_sample->m_height; ++row) {
        for (cl_int col = 0; col < m_sample->m_width / 2; ++col) {
            cl_int findex = map[row * m_sample->m_width / 2 + col];
            cl_float expected = 0.0f;
            for (cl_int frow = 0; frow < 7; ++frow) {
                for (cl_int fcol = 0; fcol < 7; ++fcol) {
                    expected += src[(row + frow) * (m_sample->m_width + 6) + col * 2 + fcol + row % 2] *
                        filter[findex * 49 + frow * 7 + fcol];
                }
            }
            float got = dst[row * m_sample->m_width / 2 + col];
            if (abs(got - expected) > Ulp(got, expected)) {
                std::cout << "Result check failed at row: " << row << ", col: " << col  << ", filter index: " << findex <<
                    ". Expected: " << expected << ", got: " << got << std::endl;
                exit(EXIT_FAILURE);
            }
        }
    }
    std::cout << "Result check pass." << std::endl;
}

void Convolution7x7::RunWithLsize(std::vector<size_t>& lsize, bool isTunning)
{
    if (!isTunning) {
        std::cout << "Using best local sizes: [" << lsize[0] << ", " << lsize[1] << "]" << std::endl;
    }

    WisePtr<cl_float> src{ static_cast<size_t>((m_sample->m_width + 6) * (m_sample->m_height + 6)) };
    WisePtr<cl_float> filter{ 7 * 7 * 10 };
    WisePtr<cl_int> map{ static_cast<size_t>(m_sample->m_width / 2 * m_sample->m_height) };
    PrepareData(src, filter, map);

    size_t lsizes[2] = {lsize[0], lsize[1]};
    size_t gsizes[2] = { static_cast<size_t>(m_sample->m_width / 8), static_cast<size_t>(m_sample->m_height) };
    cl_event event{nullptr};
    cl_int err = clEnqueueNDRangeKernel(m_sample->m_queue, m_kernel, 2, nullptr, gsizes, lsizes, 0, nullptr, &event);
    CheckClErr(err, "clEnqueueNDRangeKernel");

    err = clWaitForEvents(1, &event);
    CheckClErr(err, "clWaitForEvents");

    cl_ulong exeTime = GetProfilingInfo(event);

    err = clReleaseEvent(event);
    CheckClErr(err, "clReleaseEvent");

    if (isTunning) {
        std::initializer_list<cl_mem> initList{ m_src, m_dst, m_filter, m_map };
        ReleaseClMemObj(initList);
        m_src = nullptr;
        m_dst = nullptr;
        m_filter = nullptr;
        m_map = nullptr;
        if (exeTime < m_minExeTime) {
            m_minExeTime = exeTime;
            m_bestLsize = lsize;
        }
    }
    else {
        WisePtr<cl_float> dst(m_sample->m_width / 2 * m_sample->m_height);
        err = clEnqueueReadBuffer(m_sample->m_queue, m_dst, CL_TRUE, 0, dst.size() * sizeof(cl_float), dst.get(), 0, nullptr, nullptr);
        CheckClErr(err, "clEnqueueReadBuffer");

        err = clFinish(m_sample->m_queue);
        CheckClErr(err, "clFinish");
        ValidateResult(src, dst, filter, map);

        std::initializer_list<cl_mem> initList{ m_src, m_filter, m_map };
        ReleaseClMemObj(initList);
        m_src = nullptr;
        m_filter = nullptr;
        m_map = nullptr;
    }
}
