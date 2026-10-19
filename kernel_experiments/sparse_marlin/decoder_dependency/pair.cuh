#pragma once
#include "decoder.cuh"
__device__ __forceinline__ uint32_t unpack24_to32_high(uint32_t x) {
 uint32_t t,lo,hi;
 asm volatile("lop3.b32 %0,%1,%2,%3,0xd8;":"=r"(t):"r"(x),"r"(x>>4),"r"(0x0000fff0u));x=t;
 asm volatile("lop3.b32 %0,%1,%2,%3,0xd8;":"=r"(t):"r"(x),"r"(x>>2),"r"(0x00fc00fcu));x=t;
 asm volatile("lop3.b32 %0,%1,%2,%3,0xd8;":"=r"(t):"r"(x),"r"(x>>1),"r"(0x0e0e0e0eu));x=t;
 x&=0xeeeeeeeeu;
 asm volatile("lop3.b32 %0,%1,%2,%3,0x14;":"=r"(lo):"r"(x>>3),"r"(x>>1),"r"(x));
 asm volatile("lop3.b32 %0,%1,%2,%3,0xab;":"=r"(hi):"r"(x>>1),"r"(x),"r"(x<<1));
 asm volatile("lop3.b32 %0,%1,%2,%3,0xd8;":"=r"(t):"r"(hi),"r"(lo),"r"(0x11111111u));return t;
}
__device__ __forceinline__ uint32_t unpack_high_fused(uint32_t v) {
 uint32_t o;
 asm volatile("{.reg .b32 x,t,a,b,c,lo,hi; mov.b32 x,%1;\n"
 "shr.u32 t,x,4; lop3.b32 x,x,t,0x0000fff0,0xd8;\n"
 "shr.u32 t,x,2; lop3.b32 x,x,t,0x00fc00fc,0xd8;\n"
 "shr.u32 t,x,1; lop3.b32 x,x,t,0x0e0e0e0e,0xd8;\n"
 "and.b32 x,x,0xeeeeeeee;\n"
 "shr.u32 a,x,1; shr.u32 b,x,3; shl.b32 c,x,1;\n"
 "lop3.b32 lo,b,a,x,0x14; lop3.b32 hi,a,x,c,0xab;\n"
 "lop3.b32 %0,hi,lo,0x11111111,0xd8;}" :"=r"(o):"r"(v));return o;
}
// D=0 alignment only; 1 exact functions; 2 fused functions;
// D=3 two low decoders (different mapping); D=4 exact, sequential control.
// The sequential control uses a runtime-zero dependency, not a different codec.
template<int D> __device__ __forceinline__ void pair_decode(uint32_t &a,uint32_t &b,bool upper,uint32_t zero=0) {
 uint32_t h,l;
 if(upper) {asm("shf.r.wrap.b32 %0,%1,%2,8;":"=r"(h):"r"(a),"r"(b));l=b>>8;}
 else {h=a<<8;asm("shf.l.wrap.b32 %0,%1,%2,8;":"=r"(l):"r"(a),"r"(b));}
 if constexpr(D==0){a=h;b=l;}
 if constexpr(D==1){a=unpack24_to32_high(h);b=unpack24_to32(l);}
 if constexpr(D==2){a=unpack_high_fused(h);b=unpack24_fused<false>(l);}
 if constexpr(D==3){a=unpack24_fused<false>(h>>8);b=unpack24_fused<false>(l);}
 if constexpr(D==4){a=unpack24_to32_high(h);uint32_t t;asm volatile("and.b32 %0,%1,%2;":"=r"(t):"r"(a),"r"(zero));b=unpack24_to32(l^t);}
}
