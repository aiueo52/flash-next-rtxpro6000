// L2 access-policy / prefetch helpers for the L2-prefetch feasibility bench (task L2).
// Built by l2ctl.py:  nvcc -O3 -arch=sm_120 -Xcompiler -fPIC -shared -o libl2ctl.so l2ctl.cu -lcuda
//
// Everything here is a thin C ABI over the CUDA runtime/driver so the Python bench can
//   * query the persisting-L2 limits of the device,
//   * set the persisting-L2 carve-out and reset it,
//   * launch a "touch" kernel (one 4-byte load per 32-byte sector, results discarded
//     through a never-taken conditional store) with cudaLaunchKernelEx and an
//     access-policy window / priority launch attribute (Triton launches cannot do that),
//   * launch the same kernel over a device-side table of (ptr, bytes) ranges (the
//     per-layer prefetcher shape for Phase B),
//   * instantiate a captured cudaGraph_t with cudaGraphInstantiateFlagUseNodePriority
//     (torch does not pass that flag), launch it, and get/set kernel-node attributes.
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

struct L2Range {
  const void* ptr;
  unsigned long long bytes;
};

static char g_err[256] = "";
#define CK(x)                                                     \
  do {                                                            \
    cudaError_t _e = (x);                                         \
    if (_e != cudaSuccess) {                                      \
      snprintf(g_err, sizeof(g_err), "%s: %s", #x, cudaGetErrorString(_e)); \
      return (int)_e;                                             \
    }                                                             \
  } while (0)
#define CKD(x)                                                    \
  do {                                                            \
    CUresult _r = (x);                                            \
    if (_r != CUDA_SUCCESS) {                                     \
      const char* s = "?";                                        \
      cuGetErrorString(_r, &s);                                   \
      snprintf(g_err, sizeof(g_err), "%s: %s", #x, s);            \
      return (int)_r;                                             \
    }                                                             \
  } while (0)

extern "C" const char* l2_last_error() { return g_err; }

// ---- loads with L2 cache hints ----------------------------------------------
// MODE 0: ld.global                      (normal L2 policy)
// MODE 1: ld.global.cs                   (streaming / evict-first)
// MODE 2: ld.global.L2::cache_hint, createpolicy evict_last  (fraction 1.0)
// MODE 3: ld.global.L2::cache_hint, createpolicy evict_first (fraction 1.0)
template <int MODE>
__device__ __forceinline__ unsigned long long make_policy() {
  unsigned long long pol = 0;
  if (MODE == 2)
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
  if (MODE == 3)
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  return pol;
}

template <int MODE>
__device__ __forceinline__ unsigned ld32(const unsigned char* p, unsigned long long pol) {
  unsigned v;
  if (MODE == 0)
    asm volatile("ld.global.u32 %0, [%1];" : "=r"(v) : "l"(p));
  else if (MODE == 1)
    asm volatile("ld.global.cs.u32 %0, [%1];" : "=r"(v) : "l"(p));
  else
    asm volatile("ld.global.L2::cache_hint.u32 %0, [%1], %2;" : "=r"(v) : "l"(p), "l"(pol));
  return v;
}

// One 4-byte load per 32-byte sector over [base, base + nsec*32). Grid-strided,
// 4 independent loads in flight per thread.
template <int MODE>
__device__ __forceinline__ unsigned touch_range(const unsigned char* base, unsigned long long nsec,
                                                unsigned long long pol) {
  unsigned long long tid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
  unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
  unsigned acc = 0;
  unsigned long long i = tid;
  for (; i + 3 * stride < nsec; i += 4 * stride) {
    unsigned a = ld32<MODE>(base + i * 32, pol);
    unsigned b = ld32<MODE>(base + (i + stride) * 32, pol);
    unsigned c = ld32<MODE>(base + (i + 2 * stride) * 32, pol);
    unsigned d = ld32<MODE>(base + (i + 3 * stride) * 32, pol);
    acc += a ^ b ^ c ^ d;
  }
  for (; i < nsec; i += stride) acc += ld32<MODE>(base + i * 32, pol);
  return acc;
}

template <int MODE>
__global__ void touch_kernel(const unsigned char* base, unsigned long long nsec, unsigned* sink) {
  unsigned long long pol = make_policy<MODE>();
  unsigned acc = touch_range<MODE>(base, nsec, pol);
  if (acc == 0xFFFFFFFFu && sink) *sink = acc;  // keeps the loads alive; practically never taken
}

template <int MODE>
__global__ void touch_table_kernel(const L2Range* tab, int n, unsigned* sink) {
  unsigned long long pol = make_policy<MODE>();
  unsigned acc = 0;
  for (int r = 0; r < n; ++r) {
    const unsigned char* base = (const unsigned char*)tab[r].ptr;
    unsigned long long nsec = tab[r].bytes >> 5;
    acc += touch_range<MODE>(base, nsec, pol);
  }
  if (acc == 0xFFFFFFFFu && sink) *sink = acc;
}

// ---- device queries ---------------------------------------------------------
// out[0] L2 bytes, [1] max persisting L2 bytes, [2] max access-policy window bytes,
// [3] SM count, [4] least stream priority, [5] greatest stream priority,
// [6] current persisting limit, [7] cudaDevAttrL2CacheSize as reported by driver (cuDeviceGetAttribute)
extern "C" int l2_query(long long* out) {
  int dev = 0, v = 0;
  CK(cudaGetDevice(&dev));
  CK(cudaDeviceGetAttribute(&v, cudaDevAttrL2CacheSize, dev)); out[0] = v;
  CK(cudaDeviceGetAttribute(&v, cudaDevAttrMaxPersistingL2CacheSize, dev)); out[1] = v;
  CK(cudaDeviceGetAttribute(&v, cudaDevAttrMaxAccessPolicyWindowSize, dev)); out[2] = v;
  CK(cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, dev)); out[3] = v;
  int lo = 0, hi = 0;
  CK(cudaDeviceGetStreamPriorityRange(&lo, &hi)); out[4] = lo; out[5] = hi;
  size_t lim = 0;
  CK(cudaDeviceGetLimit(&lim, cudaLimitPersistingL2CacheSize)); out[6] = (long long)lim;
  CUdevice cdev; CKD(cuDeviceGet(&cdev, dev));
  int cv = 0; CKD(cuDeviceGetAttribute(&cv, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, cdev)); out[7] = cv;
  return 0;
}

extern "C" int l2_set_persisting(unsigned long long bytes, unsigned long long* actual) {
  CK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, (size_t)bytes));
  size_t lim = 0;
  CK(cudaDeviceGetLimit(&lim, cudaLimitPersistingL2CacheSize));
  if (actual) *actual = lim;
  return 0;
}

extern "C" int l2_reset_persisting() {
  CK(cudaCtxResetPersistingL2Cache());
  return 0;
}

extern "C" int l2_stream_window(void* stream, const void* base, unsigned long long bytes, float hit_ratio,
                     int hit_prop, int miss_prop) {
  cudaStreamAttrValue v;
  memset(&v, 0, sizeof(v));
  v.accessPolicyWindow.base_ptr = (void*)base;
  v.accessPolicyWindow.num_bytes = (size_t)bytes;
  v.accessPolicyWindow.hitRatio = hit_ratio;
  v.accessPolicyWindow.hitProp = (cudaAccessProperty)hit_prop;
  v.accessPolicyWindow.missProp = (cudaAccessProperty)miss_prop;
  CK(cudaStreamSetAttribute((cudaStream_t)stream, cudaStreamAttributeAccessPolicyWindow, &v));
  return 0;
}

// ---- launches ---------------------------------------------------------------
// kind 0: touch_kernel(base, nsec = n)      kind 1: touch_table_kernel(table, n ranges)
// mode: load policy (see above). win_on: attach an access-policy window launch attribute.
// prio_on: attach cudaLaunchAttributePriority (only honoured inside graphs instantiated
// with cudaGraphInstantiateFlagUseNodePriority, or as the stream's own priority).
extern "C" int l2_launch(void* stream, int kind, int mode, const void* p, unsigned long long n, int grid,
              int block, int win_on, const void* win_base, unsigned long long win_bytes,
              float hit_ratio, int hit_prop, int miss_prop, int prio_on, int prio, void* sink) {
  cudaLaunchConfig_t cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.gridDim = dim3(grid, 1, 1);
  cfg.blockDim = dim3(block, 1, 1);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = (cudaStream_t)stream;
  cudaLaunchAttribute attrs[2];
  memset(attrs, 0, sizeof(attrs));
  int na = 0;
  if (win_on) {
    attrs[na].id = cudaLaunchAttributeAccessPolicyWindow;
    attrs[na].val.accessPolicyWindow.base_ptr = (void*)win_base;
    attrs[na].val.accessPolicyWindow.num_bytes = (size_t)win_bytes;
    attrs[na].val.accessPolicyWindow.hitRatio = hit_ratio;
    attrs[na].val.accessPolicyWindow.hitProp = (cudaAccessProperty)hit_prop;
    attrs[na].val.accessPolicyWindow.missProp = (cudaAccessProperty)miss_prop;
    na++;
  }
  if (prio_on) {
    attrs[na].id = cudaLaunchAttributePriority;
    attrs[na].val.priority = prio;
    na++;
  }
  cfg.attrs = attrs;
  cfg.numAttrs = na;
  unsigned* s = (unsigned*)sink;
  if (kind == 0) {
    const unsigned char* b = (const unsigned char*)p;
    switch (mode) {
      case 0: CK(cudaLaunchKernelEx(&cfg, touch_kernel<0>, b, n, s)); break;
      case 1: CK(cudaLaunchKernelEx(&cfg, touch_kernel<1>, b, n, s)); break;
      case 2: CK(cudaLaunchKernelEx(&cfg, touch_kernel<2>, b, n, s)); break;
      default: CK(cudaLaunchKernelEx(&cfg, touch_kernel<3>, b, n, s)); break;
    }
  } else {
    const L2Range* t = (const L2Range*)p;
    int nr = (int)n;
    switch (mode) {
      case 0: CK(cudaLaunchKernelEx(&cfg, touch_table_kernel<0>, t, nr, s)); break;
      case 1: CK(cudaLaunchKernelEx(&cfg, touch_table_kernel<1>, t, nr, s)); break;
      case 2: CK(cudaLaunchKernelEx(&cfg, touch_table_kernel<2>, t, nr, s)); break;
      default: CK(cudaLaunchKernelEx(&cfg, touch_table_kernel<3>, t, nr, s)); break;
    }
  }
  return 0;
}

// ---- graphs -----------------------------------------------------------------
extern "C" int l2_graph_instantiate(void* graph, unsigned long long flags, void** exec_out) {
  cudaGraphExec_t e = nullptr;
  CK(cudaGraphInstantiateWithFlags(&e, (cudaGraph_t)graph, flags));
  *exec_out = (void*)e;
  return 0;
}
extern "C" int l2_graph_launch(void* exec, void* stream) {
  CK(cudaGraphLaunch((cudaGraphExec_t)exec, (cudaStream_t)stream));
  return 0;
}
extern "C" int l2_graph_exec_destroy(void* exec) {
  CK(cudaGraphExecDestroy((cudaGraphExec_t)exec));
  return 0;
}
extern "C" int l2_graph_num_nodes(void* graph, unsigned long long* n) {
  size_t k = 0;
  CK(cudaGraphGetNodes((cudaGraph_t)graph, nullptr, &k));
  *n = k;
  return 0;
}
static int get_node(void* graph, int idx, cudaGraphNode_t* out) {
  size_t k = 0;
  CK(cudaGraphGetNodes((cudaGraph_t)graph, nullptr, &k));
  if (idx < 0 || (size_t)idx >= k) { snprintf(g_err, sizeof(g_err), "node index out of range"); return -1; }
  cudaGraphNode_t* nodes = new cudaGraphNode_t[k];
  cudaError_t e = cudaGraphGetNodes((cudaGraph_t)graph, nodes, &k);
  *out = nodes[idx];
  delete[] nodes;
  CK(e);
  return 0;
}
// type: cudaGraphNodeType (0 = kernel). For kernel nodes: name (via driver), window attr.
extern "C" int l2_graph_node_info(void* graph, int idx, int* type, char* name, int name_cap, long long* win,
                       float* hit_ratio, int* prio) {
  cudaGraphNode_t node;
  int r = get_node(graph, idx, &node);
  if (r) return r;
  cudaGraphNodeType t;
  CK(cudaGraphNodeGetType(node, &t));
  *type = (int)t;
  if (name_cap > 0) name[0] = 0;
  if (t != cudaGraphNodeTypeKernel) return 0;
  CUDA_KERNEL_NODE_PARAMS kp;
  memset(&kp, 0, sizeof(kp));
  if (cuGraphKernelNodeGetParams((CUgraphNode)node, &kp) == CUDA_SUCCESS && name_cap > 0) {
    const char* nm = nullptr;
    if (kp.func && cuFuncGetName(&nm, kp.func) == CUDA_SUCCESS && nm) {
      strncpy(name, nm, name_cap - 1); name[name_cap - 1] = 0;
    } else if (kp.kern) {
      CUfunction f = nullptr;
      if (cuKernelGetFunction(&f, kp.kern) == CUDA_SUCCESS && cuFuncGetName(&nm, f) == CUDA_SUCCESS && nm) {
        strncpy(name, nm, name_cap - 1); name[name_cap - 1] = 0;
      }
    }
  }
  cudaKernelNodeAttrValue v;
  memset(&v, 0, sizeof(v));
  CK(cudaGraphKernelNodeGetAttribute(node, cudaKernelNodeAttributeAccessPolicyWindow, &v));
  win[0] = (long long)(uintptr_t)v.accessPolicyWindow.base_ptr;
  win[1] = (long long)v.accessPolicyWindow.num_bytes;
  win[2] = (long long)v.accessPolicyWindow.hitProp;
  win[3] = (long long)v.accessPolicyWindow.missProp;
  *hit_ratio = v.accessPolicyWindow.hitRatio;
  memset(&v, 0, sizeof(v));
  if (cudaGraphKernelNodeGetAttribute(node, cudaKernelNodeAttributePriority, &v) == cudaSuccess)
    *prio = v.priority;
  else { *prio = 0; cudaGetLastError(); }
  return 0;
}
extern "C" int l2_graph_node_set_window(void* graph, int idx, const void* base, unsigned long long bytes,
                             float hit_ratio, int hit_prop, int miss_prop) {
  cudaGraphNode_t node;
  int r = get_node(graph, idx, &node);
  if (r) return r;
  cudaKernelNodeAttrValue v;
  memset(&v, 0, sizeof(v));
  v.accessPolicyWindow.base_ptr = (void*)base;
  v.accessPolicyWindow.num_bytes = (size_t)bytes;
  v.accessPolicyWindow.hitRatio = hit_ratio;
  v.accessPolicyWindow.hitProp = (cudaAccessProperty)hit_prop;
  v.accessPolicyWindow.missProp = (cudaAccessProperty)miss_prop;
  CK(cudaGraphKernelNodeSetAttribute(node, cudaKernelNodeAttributeAccessPolicyWindow, &v));
  return 0;
}
extern "C" int l2_graph_node_set_priority(void* graph, int idx, int prio) {
  cudaGraphNode_t node;
  int r = get_node(graph, idx, &node);
  if (r) return r;
  cudaKernelNodeAttrValue v;
  memset(&v, 0, sizeof(v));
  v.priority = prio;
  CK(cudaGraphKernelNodeSetAttribute(node, cudaKernelNodeAttributePriority, &v));
  return 0;
}
extern "C" int l2_sync() { CK(cudaDeviceSynchronize()); return 0; }

