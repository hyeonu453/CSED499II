#pragma once
#include <stdint.h>

// User-supplied decoder, preserved exactly in its operations and LUT constants.
__device__ __forceinline__
uint32_t unpack24_to32(uint32_t x) {
    uint32_t t, s, lo, hi;
    asm volatile("lop3.b32 %0, %1, %2, %3, 0xD8;"
        : "=r"(t) : "r"(x), "r"(x << 4), "r"(0xFFFF0000u));
    x = t;
    asm volatile("lop3.b32 %0, %1, %2, %3, 0xD8;"
        : "=r"(t) : "r"(x), "r"(x << 2), "r"(0x3F003F00u));
    x = t;
    asm volatile("lop3.b32 %0, %1, %2, %3, 0xD8;"
        : "=r"(t) : "r"(x), "r"(x << 1), "r"(0x70707070u));
    x = t;
    x &= 0x77777777u;
    s = x >> 1;
    asm volatile("lop3.b32 %0, %1, %2, %1, 0x30;"
        : "=r"(lo) : "r"(x), "r"(s));
    asm volatile("lop3.b32 %0, %1, %2, %3, 0x47;"
        : "=r"(hi) : "r"(x), "r"(x << 1), "r"(x << 2));
    asm volatile("lop3.b32 %0, %1, %2, %3, 0xD8;"
        : "=r"(t) : "r"(lo), "r"(hi), "r"(0xCCCCCCCCu));
    return t;
}

__device__ __forceinline__ uint32_t spread24(uint32_t x) {
    uint32_t t;
    asm volatile("lop3.b32 %0,%1,%2,%3,0xd8;":"=r"(t):"r"(x),"r"(x<<4),"r"(0xffff0000u));x=t;
    asm volatile("lop3.b32 %0,%1,%2,%3,0xd8;":"=r"(t):"r"(x),"r"(x<<2),"r"(0x3f003f00u));x=t;
    asm volatile("lop3.b32 %0,%1,%2,%3,0xd8;":"=r"(t):"r"(x),"r"(x<<1),"r"(0x70707070u));
    return t&0x77777777u;
}

__device__ __forceinline__ uint32_t decode_nibbles(uint32_t x) {
    uint32_t lo,hi,t;x&=0x77777777u;
    asm volatile("lop3.b32 %0,%1,%2,%1,0x30;":"=r"(lo):"r"(x),"r"(x>>1));
    asm volatile("lop3.b32 %0,%1,%2,%3,0x47;":"=r"(hi):"r"(x),"r"(x<<1),"r"(x<<2));
    asm volatile("lop3.b32 %0,%1,%2,%3,0xd8;":"=r"(t):"r"(lo),"r"(hi),"r"(0xccccccccu));
    return t;
}

// Same mapping. Keeping the dependent operations inside one PTX block avoids
// frontend rewriting (x & mask) << n as (x << n) & shifted_mask.
template<bool FORCE_SHF> __device__ __forceinline__ uint32_t unpack24_fused(uint32_t input) {
    uint32_t output;
    if constexpr (!FORCE_SHF) {
        asm volatile(
            "{ .reg .b32 x,t,s,lo,hi;\n"
            "mov.b32 x,%1;\n"
            "shl.b32 t,x,4; lop3.b32 x,x,t,0xffff0000,0xd8;\n"
            "shl.b32 t,x,2; lop3.b32 x,x,t,0x3f003f00,0xd8;\n"
            "shl.b32 t,x,1; lop3.b32 x,x,t,0x70707070,0xd8;\n"
            "and.b32 x,x,0x77777777;\n"
            "shr.u32 s,x,1; lop3.b32 lo,x,s,x,0x30;\n"
            "shl.b32 s,x,1; shl.b32 t,x,2;\n"
            "lop3.b32 hi,x,s,t,0x47;\n"
            "lop3.b32 %0,lo,hi,0xcccccccc,0xd8; }"
            : "=r"(output) : "r"(input));
    } else {
        asm volatile(
            "{ .reg .b32 x,t,s,lo,hi;\n"
            "mov.b32 x,%1;\n"
            "shf.l.wrap.b32 t,0,x,4; lop3.b32 x,x,t,0xffff0000,0xd8;\n"
            "shf.l.wrap.b32 t,0,x,2; lop3.b32 x,x,t,0x3f003f00,0xd8;\n"
            "shf.l.wrap.b32 t,0,x,1; lop3.b32 x,x,t,0x70707070,0xd8;\n"
            "and.b32 x,x,0x77777777;\n"
            "shf.r.wrap.b32 s,x,0,1; lop3.b32 lo,x,s,x,0x30;\n"
            "shf.l.wrap.b32 s,0,x,1; shf.l.wrap.b32 t,0,x,2;\n"
            "lop3.b32 hi,x,s,t,0x47;\n"
            "lop3.b32 %0,lo,hi,0xcccccccc,0xd8; }"
            : "=r"(output) : "r"(input));
    }
    return output;
}
