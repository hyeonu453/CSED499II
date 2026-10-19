"""Build diagnostic ablations in isolation; never edit the original kernel."""
from pathlib import Path
import hashlib
import json

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "src/marlin/marlin_cuda_kernel_nm.cu"
GEN = ROOT / "generated"
GEN.mkdir(exist_ok=True)
original = SOURCE.read_text()
copy = """          cp_async4_stream(&sh_meta_stage[m_sh_wr_delta * i + m_sh_wr],
                           meta_ptr[i]);"""
load = """    ldsm4_m(frag_m[k % 2][0],
            &sh_m_stage[m_sh_rd_delta * (k % m_sh_iters) + m_sh_rd]);"""
assert original.count(copy) == original.count(load) == 1
files = []
declarations = []
bindings = []
for variant in ("no_gmem", "no_reg", "no_all"):
    namespace = "diag_" + variant
    text = original.replace("namespace marlin {", f"namespace {namespace} {{\nusing namespace marlin;", 1)
    if variant in ("no_gmem", "no_all"):
        text = text.replace(copy, "          ; // diagnostic: no global metadata copy")
    if variant == "no_gmem":
        marker = "  int4 *sh_m = sh_s + (stages * s_sh_stage);"
        text = text.replace(marker, marker + """
  // Uniform valid 2:4 metadata, initialized once outside the tile pipeline.
  for (int idx=threadIdx.x; idx<stages*m_sh_stage; idx+=threads)
    sh_m[idx]=make_int4(-286331154,-286331154,-286331154,-286331154);
  __syncthreads();
""")
    if variant in ("no_reg", "no_all"):
        text = text.replace(load, """    frag_m[k % 2][0][0]=0xeeeeeeeeu;
    frag_m[k % 2][1][0]=0xeeeeeeeeu;""")
    path = GEN / f"{variant}.cu"
    path.write_text(text)
    files.append(str(path))
    declarations.append(f"""namespace {namespace} {{
int marlin_cuda_2_4(const void*,const void*,const void*,void*,void*,int,int,int,
                    void*,int,int,cudaStream_t,int,int,int,int);
}}""")
    bindings.append(f"""
  m.def("{variant}", [](torch::Tensor a,torch::Tensor b,torch::Tensor meta,
                        torch::Tensor c,torch::Tensor s,torch::Tensor workspace) {{
    int k=a.size(1), n=c.size(1), batch=a.size(0), dev=a.get_device();
    int group=s.size(0)==1?-1:k/2/s.size(0);
    int err={namespace}::marlin_cuda_2_4(a.data_ptr(),b.data_ptr(),meta.data_ptr(),
      c.data_ptr(),s.data_ptr(),n,batch,k,workspace.data_ptr(),group,dev,
      at::cuda::getCurrentCUDAStream(dev),-1,-1,-1,16);
    TORCH_CHECK(err==0,"diagnostic kernel error ",err);
  }});
""")
binding = GEN / "bindings.cpp"
binding.write_text("#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n" +
                   "\n".join(declarations) + "\nPYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {\n" +
                   "\n".join(bindings) + "\n}\n")
(GEN / "provenance.json").write_text(json.dumps({"source":str(SOURCE),
    "sha256":hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    "note":"Uniform metadata diagnostics only; register allocation and scheduling may change."},indent=2)+"\n")
setup(name="metadata_diagnostics", ext_modules=[CUDAExtension("metadata_diagnostics",
      [str(binding),*files], include_dirs=[str(SOURCE.parent)],
      extra_compile_args={"cxx":["-O3"],"nvcc":["-O3","-arch=sm_86","-lineinfo","--ptxas-options=-v"]})],
      cmdclass={"build_ext":BuildExtension})
