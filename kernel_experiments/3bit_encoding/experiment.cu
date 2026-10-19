#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <random>
#include <vector>
#include <string>
#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)
using U=uint32_t;
__host__ __device__ U dec(U x){return 3u^(x|(7u+(x>>1)));}
__device__ __forceinline__ U spread(U x){ x=((x&0x00fff000)<<4)|(x&0x00000fff); x=((x&0x0fc00fc0)<<2)|(x&0x003f003f); return ((x&0x38383838)<<1)|(x&0x07070707); }
__device__ __forceinline__ U swar(U x){return (x|(((x>>1)&0x33333333)+0x77777777))^0x33333333;}
__device__ __forceinline__ U scalar8(U x){U y=0;
#pragma unroll
 for(int i=0;i<8;i++) y|=dec((x>>(3*i))&7)<<(4*i);return y;}
__device__ __forceinline__ U ld32(const U* p){U x;asm volatile("ld.global.u32 %0,[%1];":"=r"(x):"l"(p));return x;}
__device__ __forceinline__ U ld8(const unsigned char* p){U x;asm volatile("ld.global.u8 %0,[%1];":"=r"(x):"l"(p));return x;}
// All shuffles execute in all 32 lanes; only lanes 0..15 produce words.
template<int M> __device__ __forceinline__ U expand(const unsigned char* p,int lane){
 U x=0; int j=lane&15;
 if constexpr(M==0||M==1){U v=lane<12?ld32((const U*)p+lane):0;int k=(j*3)>>2;U a=__shfl_sync(0xffffffff,v,k),b=__shfl_sync(0xffffffff,v,k+1); x=__funnelshift_r(a,b,(j*24)&31);}
 if constexpr(M==2){if(lane<16){int k=(j*3)>>2;U a=ld32((const U*)p+k),b=ld32((const U*)p+min(k+1,11));x=__funnelshift_r(a,b,(j*24)&31);}}
 if constexpr(M==3){if(lane<16){const unsigned char* q=p+3*j;x=ld8(q)|(ld8(q+1)<<8)|(ld8(q+2)<<16);}}
 if constexpr(M==4){U a=0,b=0,c=0,d=0;if(lane<3)asm volatile("ld.global.v4.u32 {%0,%1,%2,%3},[%4];":"=r"(a),"=r"(b),"=r"(c),"=r"(d):"l"(p+16*lane));int k=(j*3)>>2;int s=k>>2,t=min(k+1,11)>>2;
 U a0=__shfl_sync(0xffffffff,a,s),b0=__shfl_sync(0xffffffff,b,s),c0=__shfl_sync(0xffffffff,c,s),d0=__shfl_sync(0xffffffff,d,s);
 U a1=__shfl_sync(0xffffffff,a,t),b1=__shfl_sync(0xffffffff,b,t),c1=__shfl_sync(0xffffffff,c,t),d1=__shfl_sync(0xffffffff,d,t);
 U lo=(k&2)?((k&1)?d0:c0):((k&1)?b0:a0);int h=(k+1)&3;U hi=(h&2)?((h&1)?d1:c1):((h&1)?b1:a1);x=__funnelshift_r(lo,hi,(j*24)&31);}
 // Transposed layout: byte b of each 24-bit group at p[16*b+j].
 if constexpr(M==5){if(lane<16)x=ld8(p+j)|(ld8(p+16+j)<<8)|(ld8(p+32+j)<<16);}
 // Each of 16 lanes loads an aligned 16-bit piece; 8 lanes load the tail.
 if constexpr(M==6){U a=0,b=0;if(lane<16){asm volatile("ld.global.u16 %0,[%1];":"=r"(a):"l"(p+2*lane));}if(lane<8)asm volatile("ld.global.u16 %0,[%1];":"=r"(b):"l"(p+32+2*lane));int k=(3*j)>>1;U l0=__shfl_sync(0xffffffff,a,k&15),l1=__shfl_sync(0xffffffff,b,k&15);U h0=__shfl_sync(0xffffffff,a,(k+1)&15),h1=__shfl_sync(0xffffffff,b,(k+1)&15);U lo=k<16?l0:l1,hi=k+1<16?h0:h1;x=(lo|(hi<<16))>>((j&1)*8);}
 if constexpr(M==7){U v=lane<12?ld32((const U*)p+lane):0;int k=(lane*12)>>5;U a=__shfl_sync(0xffffffff,v,k),b=__shfl_sync(0xffffffff,v,k+1);x=__funnelshift_r(a,b,(lane*12)&31);}
 if constexpr(M==8){int k=(lane*3)>>1;x=(ld8(p+k)|(ld8(p+k+1)<<8))>>((lane&1)*4);}
 if constexpr(M==9){int k=(lane*12)>>5;U a=ld32((const U*)p+k),b=ld32((const U*)p+min(k+1,11));x=__funnelshift_r(a,b,(lane*12)&31);}
 if constexpr(M>=7){x&=0xfff;x=((x&0xfc0)<<2)|(x&0x3f);x=((x&0x3838)<<1)|(x&0x707);return (x|(((x>>1)&0x3333)+0x7777))^0x3333;}
 if constexpr(M==0)return scalar8(x); else return swar(spread(x&0xffffff));
}
template<int M> __global__ void expansion(const unsigned char* __restrict__ in,U* __restrict__ out,int n){int lane=threadIdx.x&31;int w=(blockIdx.x*blockDim.x+threadIdx.x)>>5;if(w>=n)return;U y=expand<M>(in+48ull*w,lane);if constexpr(M>=7)((unsigned short*)out)[32ull*w+lane]=y;else if(lane<16)out[16ull*w+lane]=y;}
// A data-dependent next address serializes warp iterations, including GMEM load.
template<int M> __global__ void latency(const unsigned char* in,U* sink,unsigned long long* ticks,int iters){int l=threadIdx.x;const unsigned char* p=in;U y=0;__syncwarp();auto t=clock64();
#pragma unroll 1
 for(int i=0;i<iters;i++){y=expand<M>(p,l);asm volatile("":"+r"(y));__syncwarp();U z=__shfl_sync(0xffffffff,y,0);p=in+48*(z&1);}
 auto e=clock64();sink[l]=y;if(l==0)*ticks=e-t;}
__device__ __forceinline__ U dec_ptx(U x){U a,b,y;asm volatile("shr.u32 %0,%3,1;\nadd.u32 %1,%0,7;\nlop3.b32 %2,%3,%1,3,0x1e;":"=&r"(a),"=&r"(b),"=r"(y):"r"(x));return y;}
__device__ __forceinline__ U swar_ptx(U x){U a,b,c,y;asm volatile("shr.u32 %0,%4,1;\nand.b32 %1,%0,0x33333333;\nadd.u32 %2,%1,0x77777777;\nlop3.b32 %3,%4,%2,0x33333333,0x1e;":"=&r"(a),"=&r"(b),"=&r"(c),"=r"(y):"r"(x));return y;}
// feedback &7 (or &0x77777777) is measured and charged separately.
template<int M,int ILP> __global__ void micro(U* sink,unsigned long long* ticks,int iters){U x[ILP];
#pragma unroll
 for(int j=0;j<ILP;j++)x[j]=(threadIdx.x+j)&7;
 auto t=clock64();
#pragma unroll 1
 for(int i=0;i<iters;i++){
#pragma unroll 32
 for(int k=0;k<32;k++){
#pragma unroll
 for(int j=0;j<ILP;j++){
 if constexpr(M==0||M==3)x[j]=dec_ptx(x[j]);
 if constexpr(M==1||M==4)x[j]=swar_ptx(x[j]);
 if constexpr(M==2){asm volatile("mad.lo.u32 %0,%0,1664525,1013904223;":"+r"(x[j]));}
 if constexpr(M<2)asm volatile("and.b32 %0,%0,%1;":"+r"(x[j]):"n"(M==0?7:0x77777777));
 }}}
 auto e=clock64();U s=0;
#pragma unroll
 for(int j=0;j<ILP;j++)s^=x[j];
 sink[blockIdx.x*blockDim.x+threadIdx.x]=s;if(threadIdx.x==0)ticks[blockIdx.x]=e-t;}
__global__ void probe(const U* in,U* out){U x=in[threadIdx.x];out[threadIdx.x]=dec(x);}
__global__ void packed_probe(const U* in,U* out){out[threadIdx.x]=swar(spread(in[threadIdx.x]&0xffffff));}
__global__ void exhaustive(U* errors){U x=blockIdx.x*blockDim.x+threadIdx.x;if(x<(1u<<24)){if(swar(spread(x))!=scalar8(x))atomicAdd(errors,1);}}
const char* names[]={"shuffle_scalar","shuffle_swar","direct_u32","direct_u8","vector_u128","transposed_u8","coop_u16","shuffle_swar4","direct_u8_4","direct_u32_4"};
using Kernel=void(*)(const unsigned char*,U*,int);
Kernel kernels[]={expansion<0>,expansion<1>,expansion<2>,expansion<3>,expansion<4>,expansion<5>,expansion<6>,expansion<7>,expansion<8>,expansion<9>};
using Lat=void(*)(const unsigned char*,U*,unsigned long long*,int);
Lat lats[]={latency<0>,latency<1>,latency<2>,latency<3>,latency<4>,latency<5>,latency<6>,latency<7>,latency<8>,latency<9>};
float bench(Kernel k,const unsigned char* in,U* out,int n,int b,int reps){cudaEvent_t a,z;cudaStream_t stream;cudaGraph_t graph;cudaGraphExec_t exec;CK(cudaEventCreate(&a));CK(cudaEventCreate(&z));CK(cudaStreamCreate(&stream));int grid=(n+b/32-1)/(b/32);
 CK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));for(int i=0;i<reps;i++)k<<<grid,b,0,stream>>>(in,out,n);CK(cudaStreamEndCapture(stream,&graph));CK(cudaGraphInstantiate(&exec,graph,0));CK(cudaGraphLaunch(exec,stream));CK(cudaEventRecord(a,stream));for(int i=0;i<3;i++)CK(cudaGraphLaunch(exec,stream));CK(cudaEventRecord(z,stream));CK(cudaEventSynchronize(z));float ms;CK(cudaEventElapsedTime(&ms,a,z));CK(cudaGraphExecDestroy(exec));CK(cudaGraphDestroy(graph));CK(cudaStreamDestroy(stream));CK(cudaEventDestroy(a));CK(cudaEventDestroy(z));return ms/(3*reps);}

int main(int argc,char**argv){cudaDeviceProp prop;CK(cudaGetDeviceProperties(&prop,0));fprintf(stderr,"GPU=%s cc=%d%d SM=%d clock_kHz=%d memoryClock_kHz=%d L2=%d\n",prop.name,prop.major,prop.minor,prop.multiProcessorCount,prop.clockRate,prop.memoryClockRate,prop.l2CacheSize);
 bool verify=argc>1&&std::string(argv[1])=="verify";const int maxn=verify?4099:1<<20;std::vector<unsigned char> in(48ull*maxn),tr(in.size());std::vector<U> ref(16ull*maxn);std::mt19937 rng(20260930);for(int w=0;w<maxn;w++)for(int j=0;j<16;j++){U x=rng()&0xffffff;if(w<8)x=(0x249249u*w);if(w==8)x=0;if(w==9)x=0xffffff;for(int b=0;b<3;b++){in[48ull*w+3*j+b]=x>>(8*b);tr[48ull*w+16*b+j]=x>>(8*b);}U y=0;for(int c=0;c<8;c++)y|=dec((x>>(3*c))&7)<<(4*c);ref[16ull*w+j]=y;}
 unsigned char *di,*dt;U *out,*err;unsigned long long* ticks;CK(cudaMalloc(&di,in.size()));CK(cudaMalloc(&dt,in.size()));CK(cudaMalloc(&out,ref.size()*4));CK(cudaMalloc(&err,4));CK(cudaMalloc(&ticks,4096*8));CK(cudaMemcpy(di,in.data(),in.size(),cudaMemcpyHostToDevice));CK(cudaMemcpy(dt,tr.data(),tr.size(),cudaMemcpyHostToDevice));
 if(argc>1&&std::string(argv[1])=="profile"){int m=argc>2?atoi(argv[2]):1;expansion<1><<<1,32>>>(di,out,1);CK(cudaDeviceSynchronize());kernels[m]<<<8192,128>>>(m==5?dt:di,out,32768);CK(cudaDeviceSynchronize());return 0;}
 U inputs[8]={0,1,2,3,4,5,6,7},expected[8]={4,4,9,8,14,14,13,12},mapped[8];CK(cudaMemcpy(err,inputs,4,cudaMemcpyHostToDevice));U* mapin;CK(cudaMalloc(&mapin,32));CK(cudaMemcpy(mapin,inputs,32,cudaMemcpyHostToDevice));probe<<<1,8>>>(mapin,out);CK(cudaMemcpy(mapped,out,32,cudaMemcpyDeviceToHost));CK(cudaFree(mapin));for(int x=0;x<8;x++){if(mapped[x]!=expected[x]||dec(x)!=expected[x])return 4;fprintf(stderr,"map %d -> 0x%x\n",x,dec(x));}
 CK(cudaMemset(err,0,4));exhaustive<<<65536,256>>>(err);U errors;CK(cudaMemcpy(&errors,err,4,cudaMemcpyDeviceToHost));fprintf(stderr,"Exhaustive packed8: %u errors / 16777216 cases\n",errors);if(errors)return 2;
 std::vector<U> got(ref.size());for(int m=0;m<10;m++)for(int b:{32,64,128,256}){kernels[m]<<<(maxn+b/32-1)/(b/32),b>>>(m==5?dt:di,out,maxn);CK(cudaMemcpy(got.data(),out,ref.size()*4,cudaMemcpyDeviceToHost));if(got!=ref){for(size_t i=0;i<got.size();i++)if(got[i]!=ref[i]){fprintf(stderr,"FAIL %s b%d at %zu got%x ref%x\n",names[m],b,i,got[i],ref[i]);break;}return 2;}fprintf(stderr,"PASS %s block=%d codes=%d\n",names[m],b,maxn*128);}
 // partial final block and end-of-buffer loads.
 for(int m=0;m<10;m++)for(int n:{1,2,3,7,9,31,33}){kernels[m]<<<(n+7)/8,256>>>(m==5?dt:di,out,n);CK(cudaMemcpy(got.data(),out,n*64,cudaMemcpyDeviceToHost));if(!std::equal(got.begin(),got.begin()+16*n,ref.begin()))return 3;}
 if(verify){fprintf(stderr,"VERIFY PASS\n");CK(cudaFree(di));CK(cudaFree(dt));CK(cudaFree(out));CK(cudaFree(err));CK(cudaFree(ticks));return 0;}
 printf("kind,method,block,warps,trial,ms,GBps,Gcodes_s,cycles\n");
 for(int m=0;m<10;m++){cudaFuncAttributes a;CK(cudaFuncGetAttributes(&a,kernels[m]));fprintf(stderr,"ATTR %s regs=%d local=%zu shared=%zu\n",names[m],a.numRegs,a.localSizeBytes,a.sharedSizeBytes);}
 // interleave randomized trials to reduce ordering / temperature bias.
 struct Job{int m,b,n,t;};std::vector<Job> jobs;for(int n:{4096,1048576})for(int b:{32,64,128,256})for(int m=0;m<10;m++)for(int t=0;t<5;t++)jobs.push_back({m,b,n,t});std::shuffle(jobs.begin(),jobs.end(),rng);
 for(auto j:jobs){float ms=bench(kernels[j.m],j.m==5?dt:di,out,j.n,j.b,j.n==4096?100:15);printf("throughput,%s,%d,%d,%d,%.9g,%.9g,%.9g,0\n",names[j.m],j.b,j.n,j.t,ms,j.n*112.0/(ms*1e6),j.n*128.0/(ms*1e6));fflush(stdout);}
 for(int m=0;m<10;m++)for(int t=0;t<9;t++){lats[m]<<<1,32>>>(m==5?dt:di,out,ticks,4096);unsigned long long cy;CK(cudaMemcpy(&cy,ticks,8,cudaMemcpyDeviceToHost));printf("latency,%s,32,1,%d,0,0,0,%.9g\n",names[m],t,cy/4096.0);}
 #define RUN(M,I) for(int t=0;t<7;t++){micro<M,I><<<1,32>>>(out,ticks,1024);unsigned long long cy;CK(cudaMemcpy(&cy,ticks,8,cudaMemcpyDeviceToHost));printf("micro,mode%d_ilp%d,32,1,%d,0,0,0,%.9g\n",M,I,t,cy/(1024.0*32*I));}
 RUN(0,1);RUN(0,8);RUN(1,1);RUN(1,8);RUN(2,1);RUN(2,8);RUN(3,1);RUN(3,8);RUN(4,1);RUN(4,8);
 #define SAT(M) for(int t=0;t<7;t++){cudaEvent_t a,z;CK(cudaEventCreate(&a));CK(cudaEventCreate(&z));micro<M,8><<<prop.multiProcessorCount*8,128>>>(out,ticks,256);CK(cudaEventRecord(a));for(int r=0;r<10;r++)micro<M,8><<<prop.multiProcessorCount*8,128>>>(out,ticks,256);CK(cudaEventRecord(z));CK(cudaEventSynchronize(z));float ms;CK(cudaEventElapsedTime(&ms,a,z));ms/=10;double codes=prop.multiProcessorCount*8.0*128*256*32*8*(M==4?8:1);printf("micro_saturated,mode%d_ilp8,128,%d,%d,%.9g,0,%.9g,0\n",M,prop.multiProcessorCount*32,t,ms,codes/(ms*1e6));CK(cudaEventDestroy(a));CK(cudaEventDestroy(z));}
 SAT(3);SAT(4);
 CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaFree(di));CK(cudaFree(dt));CK(cudaFree(out));CK(cudaFree(err));CK(cudaFree(ticks));return 0;
}
