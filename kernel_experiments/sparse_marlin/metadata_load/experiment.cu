#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#define CK(x) do { auto e=(x); if(e!=cudaSuccess) { fprintf(stderr,"%s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1); } } while(0)
using U = uint32_t;
constexpr U TABLE[8] = {4,4,9,8,14,14,13,12};

__device__ __forceinline__ U shared_load(const U* p) {
    U v;
    U a = (U)__cvta_generic_to_shared(p);
    // PTX volatile prevents ptxas from hoisting repeated reads of the four
    // immutable stages out of the timing loop (asm volatile alone does not).
    asm volatile("ld.volatile.shared.b32 %0,[%1];" : "=r"(v) : "r"(a) : "memory");
    return v;
}

__device__ __forceinline__ U decode8(U x) {
    x &= 0xffffff;
    x = ((x & 0x00fff000u) << 4) | (x & 0x00000fffu);
    x = ((x & 0x0fc00fc0u) << 2) | (x & 0x003f003fu);
    x = ((x & 0x38383838u) << 1) | (x & 0x07070707u);
    return (x | (((x >> 1) & 0x33333333u) + 0x77777777u)) ^ 0x33333333u;
}

// 0: existing ldmatrix.x2, 1: duplicated three loads only (no decode),
// 2: duplicated three loads + complete decode, 3: only own two words + decode,
// 4: only own two words, no bit selection or decoding,
// 5: same raw words as mode 4, stored in three 16-word planes,
// 6/7: one/four scalar loads to test instruction throughput,
// 8: same raw pair as mode 5 in a padded layout, loaded by one LDS.64,
// 9/10: overlapping ldmatrix.x2 from 48 words, raw/decoded, pairing lane p,p+16.
template<int MODE> __device__ __forceinline__ uint2 load(const U* stage, int lane) {
    if constexpr (MODE == 0 || MODE == 9 || MODE == 10) {
        int row = MODE == 0 ? (lane % 16) * 4 : (lane & 7) * 4 + ((lane >> 3) & 1) * 16;
        U a = (U)__cvta_generic_to_shared(stage + row);
        uint2 r;
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1},[%2];"
                     : "=r"(r.x), "=r"(r.y) : "r"(a));
        if constexpr (MODE == 10) {
            U lo=lane<16?r.x:__funnelshift_r(r.x,r.y,16);
            U hi=lane<16?(r.y&0xffff):(r.y>>16);
            return make_uint2(decode8(lo),decode8((lo>>24)|(hi<<8)));
        }
        return r;
    } else if constexpr (MODE == 8) {
        U address=(U)__cvta_generic_to_shared(stage+2*lane);
        uint2 r;
        asm volatile("ld.volatile.shared.v2.b32 {%0,%1},[%2];"
                     : "=r"(r.x),"=r"(r.y) : "r"(address) : "memory");
        return r;
    } else if constexpr (MODE == 5 || MODE == 6 || MODE == 7) {
        int index = lane / 2 + 16 * (lane & 1);
        U a=shared_load(stage+index);
        if constexpr(MODE==6) return make_uint2(a,a);
        U b=shared_load(stage+index+16);
        if constexpr(MODE==7) {
            U c=shared_load(stage+index),d=shared_load(stage+index+16);
            return make_uint2(a+c,b+d);
        }
        return make_uint2(a,b);
    } else {
        const U* p = stage + (lane / 2) * 3;
        U lo, hi;
        if constexpr (MODE == 1 || MODE == 2) {
            U a = shared_load(p), b = shared_load(p+1), c = shared_load(p+2);
            if constexpr (MODE == 1) return make_uint2(a ^ c, b);
            // Both adjacent threads receive the SAME three 32-bit words.
            lo = (lane & 1) ? __funnelshift_r(b,c,16) : a;
            hi = (lane & 1) ? (c >> 16) : (b & 0xffff);
        } else {
            p += lane & 1;
            U a = shared_load(p), b = shared_load(p+1);
            if constexpr (MODE == 4) return make_uint2(a,b);
            lo = (lane & 1) ? __funnelshift_r(a,b,16) : a;
            hi = (lane & 1) ? (b >> 16) : (b & 0xffff);
        }
        return make_uint2(decode8(lo), decode8((lo >> 24) | (hi << 8)));
    }
}

// LATENCY adds an explicit load-to-next-address dependence shared by the warp.
// Throughput mode has four stages and independent addresses, no MMA or GMEM loop.
template<int MODE, bool LATENCY>
__global__ void metadata_load(const U* input, uint2* output,
                             unsigned long long* ticks, int iterations) {
    __shared__ __align__(16) U smem[8 * 4 * 64];
    constexpr int words = MODE == 0 || MODE == 8 ? 64 : 48;
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    U* base = smem + warp * 4 * words;
    for (int i=lane; i<4*words; i+=32) base[i]=input[i];
    __syncthreads();
    uint2 acc = make_uint2(0,0);
    U slot=0;
    unsigned long long start=clock64();
    #pragma unroll 4
    for(int it=0; it<iterations; ++it) {
        if constexpr (!LATENCY) slot=it & 3;
        uint2 r=load<MODE>(base + slot*words, lane);
        acc.x ^= r.x; acc.y += r.y;
        if constexpr (LATENCY) slot=__shfl_sync(0xffffffff,r.x,0)&3;
    }
    unsigned long long stop=clock64();
    output[blockIdx.x*blockDim.x+threadIdx.x]=acc;
    if(threadIdx.x==0) ticks[blockIdx.x]=stop-start;
}

template<int MODE> __global__ void verify(const U* input, uint2* output) {
    __shared__ __align__(16) U smem[256];
    constexpr int words=MODE==0 || MODE==8?64:48;
    for(int i=threadIdx.x;i<words*4;i+=32) smem[i]=input[i];
    __syncthreads();
    for(int s=0;s<4;s++) output[s*32+threadIdx.x]=load<MODE>(smem+s*words,threadIdx.x);
}

using Kernel=void(*)(const U*,uint2*,unsigned long long*,int);
Kernel kernels[2][11] = {
    {metadata_load<0,false>,metadata_load<1,false>,metadata_load<2,false>,metadata_load<3,false>,metadata_load<4,false>,metadata_load<5,false>,metadata_load<6,false>,metadata_load<7,false>,metadata_load<8,false>,metadata_load<9,false>,metadata_load<10,false>},
    {metadata_load<0,true>,metadata_load<1,true>,metadata_load<2,true>,metadata_load<3,true>,metadata_load<4,true>,metadata_load<5,true>,metadata_load<6,true>,metadata_load<7,true>,metadata_load<8,true>,metadata_load<9,true>,metadata_load<10,true>}
};
const char* names[]={"ldmatrix_x2","pair_3word_raw","pair_3word_decode","own_2word_decode","own_2word_raw","planar_2word_raw","planar_1load_control","planar_4load_control","padded_64bit_raw","overlap_ldmatrix_raw","overlap_ldmatrix_decode"};

int main(int argc,char** argv) {
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop,0));
    fprintf(stderr,"GPU: %s, SMs=%d, clockRate=%d kHz\n",prop.name,prop.multiProcessorCount,prop.clockRate);
    std::mt19937 rng(20261019);
    std::vector<U> dense(256), packed(192), half_packed(192);
    std::vector<uint2> expected(128);
    for(int stage=0;stage<4;stage++) {
        U codes[32][16];
        for(int lane=0;lane<32;lane++) {
            U r[2]={0,0};
            for(int i=0;i<16;i++) { codes[lane][i]=rng()%8; r[i/8]|=TABLE[codes[lane][i]]<<(4*(i%8)); }
            expected[stage*32+lane]=make_uint2(r[0],r[1]);
            // ldmatrix.x2: each group of four lanes owns one matrix row.
            for(int mat=0;mat<2;mat++) dense[stage*64+mat*32+(lane/4)*4+lane%4]=r[mat];
        }
        for(int pair=0;pair<16;pair++) for(int i=0;i<32;i++) {
            U code=codes[2*pair+i/16][i%16];
            int bit=3*i, word=stage*48+3*pair+bit/32, shift=bit%32;
            packed[word]|=code<<shift;
            if(shift>29) packed[word+1]|=code>>(32-shift);
        }
        for(int pair=0;pair<16;pair++) for(int i=0;i<32;i++) {
            U code=codes[pair+16*(i/16)][i%16];
            int bit=3*i, word=stage*48+3*pair+bit/32, shift=bit%32;
            half_packed[word]|=code<<shift;
            if(shift>29) half_packed[word+1]|=code>>(32-shift);
        }
    }
    std::vector<U> planar(192), half_planar(192);
    for(int stage=0;stage<4;stage++) for(int pair=0;pair<16;pair++) for(int word=0;word<3;word++) {
        planar[stage*48+16*word+pair]=packed[stage*48+3*pair+word];
        half_planar[stage*48+16*word+pair]=half_packed[stage*48+3*pair+word];
    }
    std::vector<U> padded(256);
    for(int stage=0;stage<4;stage++) for(int lane=0;lane<32;lane++) {
        int i=stage*48+3*(lane/2)+(lane&1);
        padded[stage*64+2*lane]=packed[i]; padded[stage*64+2*lane+1]=packed[i+1];
    }
    U *dd,*dp,*dplanar,*dpadded,*dhalf; uint2* out; unsigned long long* ticks;
    int max_blocks=prop.multiProcessorCount*4;
    CK(cudaMalloc(&dd,dense.size()*4)); CK(cudaMalloc(&dp,packed.size()*4));
    CK(cudaMemcpy(dd,dense.data(),dense.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dp,packed.data(),packed.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dplanar,planar.size()*4));
    CK(cudaMemcpy(dplanar,planar.data(),planar.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dpadded,padded.size()*4));
    CK(cudaMemcpy(dpadded,padded.data(),padded.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dhalf,half_planar.size()*4));
    CK(cudaMemcpy(dhalf,half_planar.data(),half_planar.size()*4,cudaMemcpyHostToDevice));
    const U* inputs[]={dd,dp,dp,dp,dp,dplanar,dplanar,dplanar,dpadded,dhalf,dhalf};
    CK(cudaMalloc(&out,max_blocks*256*sizeof(uint2)));
    CK(cudaMalloc(&ticks,max_blocks*sizeof(unsigned long long)));
    std::vector<uint2> got(128);
    for(int mode : {0,2,3,10}) {
        if(mode==0) verify<0><<<1,32>>>(dd,out);
        if(mode==2) verify<2><<<1,32>>>(dp,out);
        if(mode==3) verify<3><<<1,32>>>(dp,out);
        if(mode==10) verify<10><<<1,32>>>(dhalf,out);
        CK(cudaGetLastError()); CK(cudaMemcpy(got.data(),out,got.size()*sizeof(uint2),cudaMemcpyDeviceToHost));
        for(int i=0;i<128;i++) if(got[i].x!=expected[i].x || got[i].y!=expected[i].y) {
            fprintf(stderr,"VERIFY FAIL mode=%d lane=%d got=%08x,%08x expected=%08x,%08x\n",mode,i,got[i].x,got[i].y,expected[i].x,expected[i].y); return 1;
        }
    }
    fprintf(stderr,"PASS: baseline and all full decoders match expected fragments for 4 stages x 32 lanes x 16 codes.\n");
    verify<9><<<1,32>>>(dhalf,out);
    CK(cudaGetLastError()); CK(cudaMemcpy(got.data(),out,got.size()*sizeof(uint2),cudaMemcpyDeviceToHost));
    for(int stage=0;stage<4;stage++) for(int lane=0;lane<32;lane++) {
        uint2 r=got[stage*32+lane];
        if(r.x!=half_planar[stage*48+lane] || r.y!=half_planar[stage*48+lane+16]) {
            fprintf(stderr,"OVERLAP RAW VERIFY FAIL stage=%d lane=%d\n",stage,lane); return 1;
        }
    }
    fprintf(stderr,"PASS: overlapping ldmatrix raw outputs match 48-word CPU layout.\n");
    for(int mode : {4,5,6,7,8}) {
    if(mode==4) verify<4><<<1,32>>>(dp,out);
    if(mode==5) verify<5><<<1,32>>>(dplanar,out);
    if(mode==6) verify<6><<<1,32>>>(dplanar,out);
    if(mode==7) verify<7><<<1,32>>>(dplanar,out);
    if(mode==8) verify<8><<<1,32>>>(dpadded,out);
    CK(cudaGetLastError()); CK(cudaMemcpy(got.data(),out,got.size()*sizeof(uint2),cudaMemcpyDeviceToHost));
    for(int stage=0;stage<4;stage++) for(int lane=0;lane<32;lane++) {
        int i=stage*48+3*(lane/2)+(lane&1);
        uint2 r=got[stage*32+lane];
        U want_a=packed[i],want_b=packed[i+1];
        if(mode==6) want_b=want_a;
        if(mode==7) {want_a*=2;want_b*=2;}
        if(r.x!=want_a || r.y!=want_b) {
            fprintf(stderr,"RAW VERIFY FAIL mode=%d stage=%d lane=%d\n",mode,stage,lane); return 1;
        }
    }
    }
    fprintf(stderr,"PASS: raw loads and instruction-count controls match CPU references for all 4 stages x 32 lanes.\n");
    if(argc>1 && std::string(argv[1])=="verify") { CK(cudaDeviceSynchronize()); return 0; }
    if(argc>1 && std::string(argv[1])=="profile") {
        int mode=argc>2?atoi(argv[2]):0;
        if(mode<0 || mode>10) return 2;
        kernels[0][mode]<<<max_blocks,256>>>(inputs[mode],out,ticks,4096);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); return 0;
    }
    fprintf(stdout,"kind,method,blocks,threads,sample,ms,cycles_per_warp_iteration,registers,static_smem\n");
    struct Job {int latency,mode,sample;}; std::vector<Job> jobs;
    for(int l=0;l<2;l++) for(int m=0;m<11;m++) for(int s=0;s<9;s++) jobs.push_back({l,m,s});
    std::shuffle(jobs.begin(),jobs.end(),rng);
    cudaEvent_t begin,end; CK(cudaEventCreate(&begin));CK(cudaEventCreate(&end));
    for(auto job:jobs) {
        auto kernel=kernels[job.latency][job.mode]; const U* input=inputs[job.mode];
        int blocks=job.latency?1:max_blocks, threads=job.latency?32:256, iterations=4096;
        kernel<<<blocks,threads>>>(input,out,ticks,128); CK(cudaGetLastError());
        CK(cudaEventRecord(begin));
        kernel<<<blocks,threads>>>(input,out,ticks,iterations);
        CK(cudaEventRecord(end)); CK(cudaEventSynchronize(end));
        float ms; CK(cudaEventElapsedTime(&ms,begin,end));
        unsigned long long cycles; CK(cudaMemcpy(&cycles,ticks,sizeof(cycles),cudaMemcpyDeviceToHost));
        cudaFuncAttributes attr; CK(cudaFuncGetAttributes(&attr,kernel));
        printf("%s,%s,%d,%d,%d,%.9g,%.9g,%d,%zu\n",job.latency?"dependent":"throughput",names[job.mode],blocks,threads,job.sample,ms,cycles/double(iterations),attr.numRegs,attr.sharedSizeBytes);
        fflush(stdout);
    }
    CK(cudaDeviceSynchronize()); CK(cudaFree(dd));CK(cudaFree(dp));CK(cudaFree(dplanar));CK(cudaFree(dpadded));CK(cudaFree(dhalf));CK(cudaFree(out));CK(cudaFree(ticks));
    CK(cudaEventDestroy(begin)); CK(cudaEventDestroy(end));
}
