float4 Convolution7x1(__constant float* src,
    __constant float* filter,
    int width,
    int sbias,
    int4 fbias,
    int rbias)
{
    float16 pix = vload16(0, src + sbias + rbias * (width + 6));

    float4 l1 = pix.s0246;
    float4 l2 = pix.s1357;
    float4 l3 = pix.s2468;
    float4 md = pix.s3579;
    float4 r1 = pix.s468a;
    float4 r2 = pix.s579b;
    float4 r3 = pix.s68ac;

    fbias += (int4)(rbias * 7);
    float8 f1 = vload8(0, filter + fbias.x);
    float8 f2 = vload8(0, filter + fbias.y);
    float8 f3 = vload8(0, filter + fbias.z);
    float8 f4 = vload8(0, filter + fbias.w);

    float4 l1f = { f1.s0, f2.s0, f3.s0, f4.s0 };
    float4 l2f = { f1.s1, f2.s1, f3.s1, f4.s1 };
    float4 l3f = { f1.s2, f2.s2, f3.s2, f4.s2 };
    float4 mdf = { f1.s3, f2.s3, f3.s3, f4.s3 };
    float4 r1f = { f1.s4, f2.s4, f3.s4, f4.s4 };
    float4 r2f = { f1.s5, f2.s5, f3.s5, f4.s5 };
    float4 r3f = { f1.s6, f2.s6, f3.s6, f4.s6 };

    return l1 * l1f + l2 * l2f + l3 * l3f + md * mdf + r1 * r1f + r2 * r2f + r3 * r3f;
}

// src: (width + 6) * (height + 6)
// dst: width / 2 * height
// filter: 7 * 7 * 10
// map: width / 2 * height
// gsize: width / 8, height
__kernel void Convolution7x7(__constant float* src,
    __global float* dst,
    __constant float* filter,
    __constant int* map,
    int width)
{
    const int col = get_global_id(0);
    const int row = get_global_id(1);

    int4 fbias = vload4(row * width / 8 + col, map);
    fbias *= (int4)49;

    int sbias = row * (width + 6) + col * 8 + row % 2;

    float4 pixels = (float4)0.0f;
    for (int i = 0; i < 7; ++i) {
        pixels += Convolution7x1(src, filter, width, sbias, fbias, i);
    }

    vstore4(pixels, row * width / 8 + col, dst);
}
