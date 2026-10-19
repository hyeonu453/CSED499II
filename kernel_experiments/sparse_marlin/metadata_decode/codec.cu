#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include "pair.cuh"
using U=uint32_t;
// Both inverse tables contain all six valid 2:4 patterns. Other nibbles reject.
__device__ __forceinline__ U encode(U x,bool high,U* errors){
 U out=0;
 #pragma unroll
 for(int i=0;i<8;i++){
  U n=(x>>(4*i))&15,code=8;
  if(high){if(n==14)code=0;if(n==13)code=1;if(n==8)code=2;if(n==4)code=5;if(n==9)code=6;if(n==12)code=7;}
  else {if(n==12)code=0;if(n==13)code=1;if(n==14)code=2;if(n==8)code=4;if(n==9)code=5;if(n==4)code=6;}
  if(code==8){atomicAdd(errors,1u);code=0;}out|=code<<(3*i);
 }
 return out;
}
__global__ void pack_kernel(const U* in,U* out,int pairs,U* errors){
 int idx=blockIdx.x*blockDim.x+threadIdx.x;if(idx>=pairs)return;
 int group=idx/16,p=idx%16;const U* a=in+group*64;
 unsigned long long first=encode(a[p],true,errors)|(static_cast<unsigned long long>(encode(a[p+32],false,errors))<<24);
 unsigned long long second=encode(a[p+16],true,errors)|(static_cast<unsigned long long>(encode(a[p+48],false,errors))<<24);
 out[group*48+p]=U(first);out[group*48+16+p]=U(first>>32)|U(second<<16);out[group*48+32+p]=U(second>>16);
}
__global__ void unpack_kernel(const U* in,U* out,int words){
 int idx=blockIdx.x*blockDim.x+threadIdx.x;if(idx>=words/2)return;
 int group=idx/32,lane=idx%32;U a=in[group*48+lane],b=in[group*48+16+lane];
 pair_decode<2>(a,b,bool(lane&16));out[group*64+lane]=a;out[group*64+32+lane]=b;
}
torch::Tensor pack_metadata(torch::Tensor x){
 TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type()==torch::kInt16 && x.numel()%128==0,"expected CUDA contiguous int16 metadata, multiple of 256 bytes");
 auto options=x.options().dtype(torch::kInt32);auto y=torch::empty({x.numel()*3/8},options);auto errors=torch::zeros({1},options);
 int pairs=x.numel()/8;pack_kernel<<<(pairs+255)/256,256,0,at::cuda::getCurrentCUDAStream(x.get_device())>>>(reinterpret_cast<const U*>(x.data_ptr()),reinterpret_cast<U*>(y.data_ptr()),pairs,reinterpret_cast<U*>(errors.data_ptr()));
 C10_CUDA_KERNEL_LAUNCH_CHECK();TORCH_CHECK(errors.item<int>()==0,"metadata contains unsupported 2:4 nibbles");return y;
}
torch::Tensor unpack_metadata(torch::Tensor x){
 TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type()==torch::kInt32 && x.numel()%48==0,"expected compressed metadata");
 auto y=torch::empty({x.numel()*8/3},x.options().dtype(torch::kInt16));int words=y.numel()/2;
 unpack_kernel<<<(words/2+255)/256,256,0,at::cuda::getCurrentCUDAStream(x.get_device())>>>(reinterpret_cast<const U*>(x.data_ptr()),reinterpret_cast<U*>(y.data_ptr()),words);
 C10_CUDA_KERNEL_LAUNCH_CHECK();return y;
}
