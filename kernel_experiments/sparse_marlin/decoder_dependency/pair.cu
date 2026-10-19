#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>
#include "pair.cuh"
using U=uint32_t;
#define CK(x) do{auto ck_error=(x);if(ck_error!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(ck_error));exit(1);}}while(0)
__device__ __constant__ U hm[8]={14,13,8,13,3,4,9,12},lm[8]={12,13,14,10,8,9,4,0};
__global__ void verify_pair(U* error){
 U x=blockIdx.x*blockDim.x+threadIdx.x,hr=0,lr=0;
 for(int i=0;i<8;i++){hr|=hm[(x>>(3*i))&7]<<(4*i);lr|=lm[(x>>(3*i))&7]<<(4*i);}
 if(unpack24_to32_high((x<<8)|(x&255))!=hr || unpack_high_fused(x<<8)!=hr)atomicAdd(error,1u);
 // Construct both 48-bit payloads with omitted bits deliberately nonzero.
 for(int upper=0;upper<2;upper++)for(int d=0;d<5;d++){
  U y=(x*1664525u+1013904223u)&0xffffff,ref=0;
  for(int i=0;i<8;i++)ref|=lm[(y>>(3*i))&7]<<(4*i);
  U a,b;
  if(!upper){a=x|(y<<24);b=(y>>8)|0xcafe0000u;}
  else {a=0xbeefu|(x<<16);b=(x>>16)|(y<<8);}
  if(d==0)pair_decode<0>(a,b,upper);
  if(d==1)pair_decode<1>(a,b,upper);
  if(d==2)pair_decode<2>(a,b,upper);
  if(d==3)pair_decode<3>(a,b,upper);
  if(d==4)pair_decode<4>(a,b,upper);
  if(d==0 ? ((a&0xffffff00u)!=(x<<8)|| (b&0xffffff)!=y) : (a!=(d==3?lr:hr)||b!=ref))atomicAdd(error+1,1u);
 }
}
template<int D,int I,int Layout> __global__ void pair_chain(U* out,unsigned long long* ticks,int iters,U zero){
 U a[I],b[I];bool upper=Layout==2?bool(threadIdx.x&16):bool(Layout);
 #pragma unroll
 for(int c=0;c<I;c++){a[c]=threadIdx.x*1234567u+c*13579u;b[c]=blockIdx.x*8761u+c*777u+123;}
 __syncwarp();auto start=clock64();
 #pragma unroll 1
 for(int i=0;i<iters;i++){
  #pragma unroll
  for(int r=0;r<16;r++){
   #pragma unroll
   for(int c=0;c<I;c++){
    pair_decode<D>(a[c],b[c],upper,zero);
    // Join both outputs before the next iteration: measures pair completion.
    asm volatile("xor.b32 %0,%1,%2;":"=r"(a[c]):"r"(a[c]),"r"(b[c]));
   }
  }
 }
 auto end=clock64();U sum=0;
 #pragma unroll
 for(int c=0;c<I;c++)sum^=a[c]^b[c];
 out[blockIdx.x*blockDim.x+threadIdx.x]=sum;
 if(threadIdx.x==0)ticks[blockIdx.x]=end-start;
}
using K=void(*)(U*,unsigned long long*,int,U);
#define R(D,L) {pair_chain<D,1,L>,pair_chain<D,2,L>,pair_chain<D,4,L>,pair_chain<D,8,L>}
#define DL(D) {R(D,0),R(D,1),R(D,2)}
K ks[5][3][4]={DL(0),DL(1),DL(2),DL(3),DL(4)};
int main(int argc,char**argv){
 cudaDeviceProp p;CK(cudaGetDeviceProperties(&p,0));int blocks=p.multiProcessorCount*8;
 U *out,*err;unsigned long long*t;CK(cudaMalloc(&out,blocks*128*4));CK(cudaMalloc(&err,8));CK(cudaMalloc(&t,blocks*8));
 if(argc>1 && std::string(argv[1])=="profile"){
  int d=atoi(argv[2]),i=atoi(argv[3]),l=atoi(argv[4]);if(d<0||d>4||i<0||i>3||l<0||l>2)return 2;
  ks[d][l][i]<<<blocks,128>>>(out,t,256,0);CK(cudaDeviceSynchronize());return 0;
 }
 CK(cudaMemset(err,0,8));verify_pair<<<65536,256>>>(err);U e[2];CK(cudaMemcpy(e,err,8,cudaMemcpyDeviceToHost));
 fprintf(stderr,"%s: exhaustive high=%u pair=%u mismatches\n",p.name,e[0],e[1]);if(e[0]||e[1])return 1;
 if(argc>1 && std::string(argv[1])=="verify")return 0;
 struct J{int d,l,i,k,s;};std::vector<J> jobs;
 for(int d=0;d<5;d++)for(int l=0;l<3;l++)for(int i=0;i<4;i++)for(int k=0;k<2;k++)for(int s=0;s<7;s++)jobs.push_back({d,l,i,k,s});
 std::mt19937 rng(20261020);std::shuffle(jobs.begin(),jobs.end(),rng);
 cudaEvent_t a,b;CK(cudaEventCreate(&a));CK(cudaEventCreate(&b));
 puts("decoder,layout,pairs,kind,sample,us,cycles_per_pair,registers,local_bytes");
 for(auto j:jobs){auto f=ks[j.d][j.l][j.i];int nb=j.k?blocks:1,nt=j.k?128:32;
 f<<<nb,nt>>>(out,t,32,0);CK(cudaEventRecord(a));f<<<nb,nt>>>(out,t,256,0);CK(cudaEventRecord(b));CK(cudaEventSynchronize(b));
 float ms;CK(cudaEventElapsedTime(&ms,a,b));unsigned long long c;CK(cudaMemcpy(&c,t,8,cudaMemcpyDeviceToHost));cudaFuncAttributes at;CK(cudaFuncGetAttributes(&at,f));
 printf("%d,%d,%d,%s,%d,%.9g,%.9g,%d,%zu\n",j.d,j.l,1<<j.i,j.k?"saturated":"single_warp",j.s,ms*1000,c/double(4096*(1<<j.i)),at.numRegs,at.localSizeBytes);fflush(stdout);
 }
 CK(cudaGetLastError());CK(cudaDeviceSynchronize());
}
