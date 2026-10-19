"""Generate isolated, compressed-metadata Sparse-Marlin variants."""
from pathlib import Path
import hashlib,json
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension,CUDAExtension
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'src/marlin/marlin_cuda_kernel_nm.cu'
GEN=ROOT/'generated';GEN.mkdir(exist_ok=True)
original=SOURCE.read_text()
load='''    ldsm4_m(frag_m[k % 2][0],
            &sh_m_stage[m_sh_rd_delta * (k % m_sh_iters) + m_sh_rd]);'''
files=[];decls=[];bindings=[]
for variant in ('fetch','matmul'):
 ns='decode_'+variant
 text=original.replace('#include "common/mma.h"','#include "common/mma.h"\n#include "pair.cuh"')
 text=text.replace('namespace marlin {',f'namespace {ns} {{\nusing namespace marlin;',1)
 text=text.replace('int m_gl_stride = 2 * prob_n / 8;','int m_gl_stride = 3 * prob_n / 16;')
 text=text.replace('(16 * thread_n_blocks) / 4;','(16 * thread_n_blocks) * 3 / 16;')
 text=text.replace('constexpr int m_sh_wr_delta = threads / 2;','constexpr int m_sh_wr_delta = threads * 3 / 8;')
 text=text.replace('bool m_sh_wr_pred = threadIdx.x < m_sh_wr_delta;','bool m_sh_wr_pred = threadIdx.x < m_sh_stage;')
 assert text.count(load)==1
 text=text.replace(load,'''    // Two overlapping 128-byte matrices within a 192-byte warp payload.
    int lane = threadIdx.x % 32;
    int row_word = (lane & 7) * 4 + ((lane >> 3) & 1) * 16;
    const uint32_t* compressed = reinterpret_cast<const uint32_t*>(sh_m_stage)
                               + (threadIdx.x / 32) * 48 + row_word;
    ldsm4_m(frag_m[k % 2][0], compressed);'''+('''
    pair_decode<2>(frag_m[k % 2][0][0],frag_m[k % 2][1][0],bool(lane & 16));''' if variant=='fetch' else ''))
 if variant=='matmul':
  text=text.replace('auto matmul = [&](int k) {','''auto matmul = [&](int k) {
    pair_decode<2>(frag_m[k % 2][0][0],frag_m[k % 2][1][0],bool(threadIdx.x & 16));''')
 # The compressed stage is 96 int4s in all instantiated shapes (128 originally).
 # Pointer traversal continues to use the original stripe algorithm and scaled strides.
 path=GEN/f'{variant}.cu';path.write_text(text);files.append(str(path))
 decls.append(f'''namespace {ns} {{int marlin_cuda_2_4(const void*,const void*,const void*,void*,void*,int,int,int,void*,int,int,cudaStream_t,int,int,int,int);}}''')
 bindings.append(f'''m.def("{variant}",[](torch::Tensor a,torch::Tensor b,torch::Tensor meta,torch::Tensor c,torch::Tensor s,torch::Tensor w){{
 TORCH_CHECK(a.is_cuda() && a.scalar_type()==torch::kFloat16 && a.is_contiguous(),"a must be contiguous CUDA half");
 int k=a.size(1), n=c.size(1), batch=a.size(0),dev=a.get_device();
 TORCH_CHECK(n%256==0 && k%128==0 && meta.scalar_type()==torch::kInt32 && meta.is_contiguous() && meta.numel()==n*k*3/128,"unsupported compressed metadata shape");
 int group=s.size(0)==1?-1:k/2/s.size(0);
 int err={ns}::marlin_cuda_2_4(a.data_ptr(),b.data_ptr(),meta.data_ptr(),c.data_ptr(),s.data_ptr(),n,batch,k,w.data_ptr(),group,dev,at::cuda::getCurrentCUDAStream(dev),-1,-1,-1,16);
 TORCH_CHECK(err==0,"kernel shape error ",err);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
 }});''')
binding=GEN/'bindings.cpp'
binding.write_text('''#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
torch::Tensor pack_metadata(torch::Tensor);
torch::Tensor unpack_metadata(torch::Tensor);
'''+ '\n'.join(decls)+'\nPYBIND11_MODULE(TORCH_EXTENSION_NAME,m){\nm.def("pack",&pack_metadata);m.def("unpack",&unpack_metadata);\n'+'\n'.join(bindings)+'\n}\n')
(GEN/'provenance.json').write_text(json.dumps({'source':str(SOURCE),'sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'codec':'pair_decode<2>; per-word high/low inverse mappings; 192B per warp'},indent=2)+'\n')
setup(name='metadata_decode',ext_modules=[CUDAExtension('metadata_decode',[str(binding),str(ROOT/'codec.cu'),*files],include_dirs=[str(SOURCE.parent),str(ROOT.parent/'decoder_dependency')],extra_compile_args={'cxx':['-O3'],'nvcc':['-O3','-std=c++17','-arch=sm_86','-lineinfo','--ptxas-options=-v']})],cmdclass={'build_ext':BuildExtension})
