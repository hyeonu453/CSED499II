#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>
#include "decoder.cuh"

using U=uint32_t;
#define CK(x) do{auto check_error=(x);if(check_error!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(check_error));exit(1);}}while(0)
__device__ __constant__ U truth[8]={0xc,0xd,0xe,0xa,0x8,0x9,0x4,0x0};

template<int D> __device__ __forceinline__ U apply(U x) {
    if constexpr(D==0)return unpack24_to32(x);
    if constexpr(D==1)return spread24(x);
    if constexpr(D==2)return decode_nibbles(x);
    if constexpr(D==3){U y;asm volatile("shf.r.wrap.b32 %0,%1,%1,1;":"=r"(y):"r"(x));return y;}
    if constexpr(D==4)return unpack24_fused<false>(x);
    if constexpr(D==5)return unpack24_fused<true>(x);
}

__global__ void verify(U* errors) {
    U x=blockIdx.x*blockDim.x+threadIdx.x;
    if(x>0xffffff)return;
    U ref=0,spread=0;
    #pragma unroll
    for(int i=0;i<8;i++) {U c=(x>>(3*i))&7;ref|=truth[c]<<(4*i);spread|=c<<(4*i);}
    if(unpack24_to32(x)!=ref)atomicAdd(errors,1u);
    U garbage=((x*1664525u+1013904223u)&0xff)<<24;
    if(unpack24_to32(x|garbage)!=ref)atomicAdd(errors+1,1u);
    if(spread24(x)!=spread)atomicAdd(errors+2,1u);
    if(decode_nibbles(spread)!=ref)atomicAdd(errors+3,1u);
    if(unpack24_fused<false>(x)!=ref || unpack24_fused<false>(x|garbage)!=ref)atomicAdd(errors+4,1u);
    if(unpack24_fused<true>(x)!=ref || unpack24_fused<true>(x|garbage)!=ref)atomicAdd(errors+5,1u);
}

template<int D> __global__ void probe(const U* in,U* out) {
    U x=in[threadIdx.x];out[threadIdx.x]=apply<D>(x);
}

template<int D,int ILP> __global__ void decoder_chain(U* out,unsigned long long* cycles,int iterations) {
    U x[ILP];
    #pragma unroll
    for(int c=0;c<ILP;c++)x[c]=(threadIdx.x*1234567u+blockIdx.x*8761u+c*13579u)&0xffffff;
    __syncwarp();
    unsigned long long begin=clock64();
    #pragma unroll 1
    for(int i=0;i<iterations;i++) {
        #pragma unroll
        for(int repeat=0;repeat<16;repeat++) {
            #pragma unroll
            for(int c=0;c<ILP;c++)x[c]=apply<D>(x[c]);
        }
    }
    unsigned long long end=clock64();
    U checksum=0;
    #pragma unroll
    for(int c=0;c<ILP;c++)checksum^=x[c];
    out[blockIdx.x*blockDim.x+threadIdx.x]=checksum;
    if(threadIdx.x==0)cycles[blockIdx.x]=end-begin;
}

using Kernel=void(*)(U*,unsigned long long*,int);
#define ROW(D) {decoder_chain<D,1>,decoder_chain<D,2>,decoder_chain<D,4>,decoder_chain<D,8>}
Kernel kernels[6][4]={ROW(0),ROW(1),ROW(2),ROW(3),ROW(4),ROW(5)};
const char* names[]={"provided_full","spreading_only","nibble_decode_only","rotate_control","fused_ptx","forced_shf"};

int main(int argc,char** argv) {
    cudaDeviceProp prop;CK(cudaGetDeviceProperties(&prop,0));
    fprintf(stderr,"GPU=%s SM=%d cc=%d%d\n",prop.name,prop.multiProcessorCount,prop.major,prop.minor);
    int maxblocks=prop.multiProcessorCount*8;
    U *out,*errors,*in;unsigned long long* ticks;
    CK(cudaMalloc(&out,maxblocks*128*sizeof(U)));CK(cudaMalloc(&errors,6*sizeof(U)));
    CK(cudaMalloc(&in,32*sizeof(U)));CK(cudaMalloc(&ticks,maxblocks*sizeof(unsigned long long)));
    U hostin[32];for(int i=0;i<32;i++)hostin[i]=i<8?i:(0xffffffu-12345u*i);
    CK(cudaMemcpy(in,hostin,sizeof(hostin),cudaMemcpyHostToDevice));
    if(argc>1 && std::string(argv[1])=="profile") {
        int d=argc>2?atoi(argv[2]):0, ilp=argc>3?atoi(argv[3]):2;
        if(d<0||d>5||ilp<1||ilp>8||(ilp&(ilp-1)))return 2;
        int index=ilp==1?0:ilp==2?1:ilp==4?2:3;
        kernels[d][index]<<<maxblocks,128>>>(out,ticks,256);
        CK(cudaGetLastError());CK(cudaDeviceSynchronize());return 0;
    }
    CK(cudaMemset(errors,0,6*sizeof(U)));
    verify<<<65536,256>>>(errors);
    U e[6];CK(cudaGetLastError());CK(cudaMemcpy(e,errors,sizeof(e),cudaMemcpyDeviceToHost));
    fprintf(stderr,"Exhaustive 2^24 inputs: full=%u, high-garbage=%u, spread=%u, nibble=%u, fused=%u, shf=%u mismatches\n",e[0],e[1],e[2],e[3],e[4],e[5]);
    if(e[0]||e[1]||e[2]||e[3]||e[4]||e[5])return 1;
    probe<0><<<1,32>>>(in,out);U map[32];CK(cudaMemcpy(map,out,sizeof(map),cudaMemcpyDeviceToHost));
    for(int i=0;i<8;i++)fprintf(stderr,"code %d -> 0x%x (word=%08x)\n",i,map[i]&15,map[i]);
    probe<1><<<1,32>>>(in,out);probe<2><<<1,32>>>(in,out);CK(cudaDeviceSynchronize());
    probe<4><<<1,32>>>(in,out);probe<5><<<1,32>>>(in,out);CK(cudaDeviceSynchronize());
    if(argc>1 && std::string(argv[1])=="verify")return 0;
    struct Job {int d,index,kind,sample;};std::vector<Job> jobs;
    for(int d=0;d<6;d++)for(int index=0;index<4;index++)for(int kind=0;kind<2;kind++)for(int s=0;s<7;s++)jobs.push_back({d,index,kind,s});
    std::mt19937 rng(20261019);std::shuffle(jobs.begin(),jobs.end(),rng);
    cudaEvent_t a,b;CK(cudaEventCreate(&a));CK(cudaEventCreate(&b));
    printf("kind,decoder,ilp,blocks,threads,sample,us,cycles_per_call,registers,local_bytes\n");
    for(auto j:jobs) {
        int blocks=j.kind?maxblocks:1,threads=j.kind?128:32,iterations=256,ilp=1<<j.index;
        auto k=kernels[j.d][j.index];k<<<blocks,threads>>>(out,ticks,32);
        CK(cudaGetLastError());CK(cudaEventRecord(a));k<<<blocks,threads>>>(out,ticks,iterations);
        CK(cudaEventRecord(b));CK(cudaEventSynchronize(b));
        float ms;CK(cudaEventElapsedTime(&ms,a,b));
        unsigned long long c;CK(cudaMemcpy(&c,ticks,sizeof(c),cudaMemcpyDeviceToHost));
        cudaFuncAttributes attr;CK(cudaFuncGetAttributes(&attr,k));
        printf("%s,%s,%d,%d,%d,%d,%.9g,%.9g,%d,%zu\n",j.kind?"saturated":"single_warp",names[j.d],ilp,blocks,threads,j.sample,ms*1000,c/double(iterations*16*ilp),attr.numRegs,attr.localSizeBytes);
        fflush(stdout);
    }
    CK(cudaDeviceSynchronize());CK(cudaFree(out));CK(cudaFree(errors));CK(cudaFree(in));CK(cudaFree(ticks));
    CK(cudaEventDestroy(a));CK(cudaEventDestroy(b));
}
