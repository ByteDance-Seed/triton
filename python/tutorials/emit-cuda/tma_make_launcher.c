// constants: {(6,): 128, (7,): 128, (8,): 32, (9,): 8}
// signature: {'a_desc': 'tensordesc<fp16[128, 32]>', 'b_desc': 'tensordesc<fp16[32, 128]>', 'c_desc': 'tensordesc<fp16[128, 128]>', 'M': 'i32', 'N': 'i32', 'K': 'i32', 'BLOCK_SIZE_M': 'constexpr', 'BLOCK_SIZE_N': 'constexpr', 'BLOCK_SIZE_K': 'constexpr', 'GROUP_SIZE_M': 'constexpr'}
// tensordesc_meta: [{'swizzle': 2, 'elem_size': 2, 'elem_type': 6, 'block_size': [128, 32], 'fp4_padded': False}, {'swizzle': 3, 'elem_size': 2, 'elem_type': 6, 'block_size': [32, 64], 'fp4_padded': False}, {'swizzle': 3, 'elem_size': 2, 'elem_type': 6, 'block_size': [128, 64], 'fp4_padded': False}]


#include "cuda.h"
#include <dlfcn.h>
#include <stdbool.h>
#include <stdlib.h>
#define PY_SSIZE_T_CLEAN
#include <Python.h>

typedef struct {
  PyObject_HEAD;
  _Alignas(128) CUtensorMap tensorMap;
} PyCUtensorMapObject;

static inline void gpuAssert(CUresult code, const char *file, int line)
{
   if (code != CUDA_SUCCESS)
   {
      const char* prefix = "Triton Error [CUDA]: ";
      const char* str;
      cuGetErrorString(code, &str);
      char err[1024] = {0};
      strcat(err, prefix);
      strcat(err, str);
      PyGILState_STATE gil_state;
      gil_state = PyGILState_Ensure();
      PyErr_SetString(PyExc_RuntimeError, err);
      PyGILState_Release(gil_state);
   }
}

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

typedef CUresult (*cuLaunchKernelEx_t)(const CUlaunchConfig* config, CUfunction f, void** kernelParams, void** extra);

static cuLaunchKernelEx_t getLaunchKernelExHandle() {
  // Open the shared library
  void* handle = dlopen("libcuda.so.1", RTLD_LAZY);
  if (!handle) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to open libcuda.so.1");
    return NULL;
  }
  // Clear any existing error
  dlerror();
  cuLaunchKernelEx_t cuLaunchKernelExHandle = (cuLaunchKernelEx_t)dlsym(handle, "cuLaunchKernelEx");
  // Check for errors
  const char *dlsym_error = dlerror();
  if (dlsym_error) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to retrieve cuLaunchKernelEx from libcuda.so.1");
    return NULL;
  }
  return cuLaunchKernelExHandle;
}

static void _launch(int gridX, int gridY, int gridZ, int num_warps, int num_ctas, int launch_cooperative_grid, int launch_pdl, int clusterDimX, int clusterDimY, int clusterDimZ, int shared_memory, CUstream stream, CUfunction function, CUdeviceptr global_scratch, CUdeviceptr profile_scratch, CUtensorMap arg0, int32_t arg1, int32_t arg2, int64_t arg3, int64_t arg4, CUtensorMap arg5, int32_t arg6, int32_t arg7, int64_t arg8, int64_t arg9, CUtensorMap arg10, int32_t arg11, int32_t arg12, int64_t arg13, int64_t arg14, int32_t arg15, int32_t arg16, int32_t arg17) {
  void *params[] = { &arg0, &arg1, &arg2, &arg3, &arg4, &arg5, &arg6, &arg7, &arg8, &arg9, &arg10, &arg11, &arg12, &arg13, &arg14, &arg15, &arg16, &arg17, &global_scratch, &profile_scratch };
  if (gridX*gridY*gridZ > 0) {
    // 4 attributes that we can currently pass maximum
    CUlaunchAttribute launchAttr[4];
    static cuLaunchKernelEx_t cuLaunchKernelExHandle = NULL;
    if (cuLaunchKernelExHandle == NULL) {
      cuLaunchKernelExHandle = getLaunchKernelExHandle();
    }
    CUlaunchConfig config;
    config.gridDimX = gridX;
    config.gridDimY = gridY;
    config.gridDimZ = gridZ;

    if (num_ctas != 1) {
      config.gridDimX *= clusterDimX;
      config.gridDimY *= clusterDimY;
      config.gridDimZ *= clusterDimZ;
    }

    config.blockDimX = 32 * num_warps;
    config.blockDimY = 1;
    config.blockDimZ = 1;
    config.sharedMemBytes = shared_memory;
    config.hStream = stream;
    config.attrs = launchAttr;
    int num_attrs = 0;

    if (launch_pdl != 0) {
      CUlaunchAttribute pdlAttr = { .id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION, .value = 1};
      launchAttr[num_attrs] = pdlAttr;
      ++num_attrs;
    }

    if (launch_cooperative_grid != 0) {
      CUlaunchAttribute coopAttr = { .id = CU_LAUNCH_ATTRIBUTE_COOPERATIVE, .value = 1};
      launchAttr[num_attrs] = coopAttr;
      ++num_attrs;
    }

    if (num_ctas != 1) {
      CUlaunchAttribute clusterAttr = {};
      clusterAttr.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
      clusterAttr.value.clusterDim.x = clusterDimX;
      clusterAttr.value.clusterDim.y = clusterDimY;
      clusterAttr.value.clusterDim.z = clusterDimZ;
      launchAttr[num_attrs] = clusterAttr;
      ++num_attrs;

      CUlaunchAttribute clusterSchedulingAttr = {};
      clusterSchedulingAttr.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
      clusterSchedulingAttr.value.clusterSchedulingPolicyPreference = CU_CLUSTER_SCHEDULING_POLICY_SPREAD;
      launchAttr[num_attrs] = clusterSchedulingAttr;
      ++num_attrs;
    }

    config.numAttrs = num_attrs;

    CUDA_CHECK(cuLaunchKernelExHandle(&config, function, params, 0));
  }
}

typedef struct _DevicePtrInfo {
    CUdeviceptr dev_ptr;
    bool valid;
} DevicePtrInfo;

static PyObject* data_ptr_str = NULL;
static PyObject* py_tensor_map_type = NULL;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {
    ptr_info.dev_ptr = PyLong_AsUnsignedLongLong(obj);
    return ptr_info;
  }
  if (obj == Py_None) {
    // valid nullptr
    return ptr_info;
  }
  PyObject *ret = PyObject_CallMethodNoArgs(obj, data_ptr_str);
  if (!ret) {
    PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
    ptr_info.valid = false;
    goto cleanup;
  }
  if (!PyLong_Check(ret)) {
    PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
    ptr_info.valid = false;
    goto cleanup;
  }
  ptr_info.dev_ptr = PyLong_AsUnsignedLongLong(ret);
  if(!ptr_info.dev_ptr)
    return ptr_info;
  uint64_t dev_ptr;
  int status = cuPointerGetAttribute(&dev_ptr, CU_POINTER_ATTRIBUTE_DEVICE_POINTER, ptr_info.dev_ptr);
  if (status == CUDA_ERROR_INVALID_VALUE) {
      PyErr_Format(PyExc_ValueError,
                   "Pointer argument (at %d) cannot be accessed from Triton (cpu tensor?)", idx);
      ptr_info.valid = false;
  } else if (status != CUDA_SUCCESS) {
      CUDA_CHECK(status);  // Catch any other cuda API errors
      ptr_info.valid = false;
  }
  ptr_info.dev_ptr = dev_ptr;
cleanup:
  Py_XDECREF(ret);
  return ptr_info;

}

static inline CUtensorMap* getTmaDesc(PyObject *obj) {
  if (sizeof(CUtensorMap*) != 8) {
    PyErr_SetString(PyExc_SystemError, "getTmaDesc() requires 64-bit compilation");
    return NULL;
  }

if (Py_TYPE(obj) != (PyTypeObject*)py_tensor_map_type) {
    PyErr_Format(PyExc_TypeError, "object must be of type PyCUtensorMap, got %s", Py_TYPE(obj)->tp_name);
    return NULL;
}

  CUtensorMap* map = &((PyCUtensorMapObject*)obj)->tensorMap;
  uintptr_t align_128 = (uintptr_t)map & (128 - 1);
  if (align_128 != 0) {
    PyErr_Format(PyExc_ValueError, "CUtensorMap must be aligned to 128B, but got (&map) mod 128 = %ld", align_128);
    return NULL;
  }
  return map;
}

static void ensureCudaContext() {
  CUcontext pctx;
  CUDA_CHECK(cuCtxGetCurrent(&pctx));
  if (!pctx) {
    // Ensure device context.
    CUdevice device;
    CUDA_CHECK(cuDeviceGet(&device, 0));
    CUDA_CHECK(cuDevicePrimaryCtxRetain(&pctx, device));
    CUDA_CHECK(cuCtxSetCurrent(pctx));
  }
}

static uint16_t pack_fp16(double f) {
    uint16_t result;
    // from https://github.com/python/pythoncapi-compat
#if 0x030600B1 <= PY_VERSION_HEX && PY_VERSION_HEX <= 0x030B00A1 && !defined(PYPY_VERSION)
    _PyFloat_Pack2(f, (unsigned char*)&result, 1);
#else
    PyFloat_Pack2(f, (unsigned char*)&result, 1);
#endif
    return result;
}

static uint16_t pack_bf16(double f) {
    float f32 = (float)f;
    uint32_t u32 = *(uint32_t*)&f32;
    return (uint16_t)(u32 >> 16);
}

static uint32_t pack_fp32(double f) {
    float f32 = (float)f;
    return *(uint32_t*)&f32;
}

static uint64_t pack_fp64(double f) {
    return *(uint64_t*)&f;
}

static PyObject* launch(PyObject* self, PyObject* args) {
  // ensure cuda context is valid before calling any CUDA APIs, e.g. before getPointer calls cuPointerGetAttributes
  ensureCudaContext();

  int gridX, gridY, gridZ;
  uint64_t _stream;
  uint64_t _function;
  int launch_cooperative_grid;
  int launch_pdl;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *kernel_metadata = NULL;
  PyObject *launch_metadata = NULL;
  PyObject *global_scratch_obj = NULL;
  PyObject *profile_scratch_obj = NULL;
  PyObject* _arg0;
  int32_t _arg1;
  int32_t _arg2;
  int64_t _arg3;
  int64_t _arg4;
  PyObject* _arg5;
  int32_t _arg6;
  int32_t _arg7;
  int64_t _arg8;
  int64_t _arg9;
  PyObject* _arg10;
  int32_t _arg11;
  int32_t _arg12;
  int64_t _arg13;
  int64_t _arg14;
  int32_t _arg15;
  int32_t _arg16;
  int32_t _arg17;
  PyObject* _arg18;
  PyObject* _arg19;
  PyObject* _arg20;
  PyObject* _arg21;
  if(!PyArg_ParseTuple(args, "iiiKKppOOOOOOOiiLLOiiLLOiiLLiiiOOOO", &gridX, &gridY, &gridZ,
                                           &_stream, &_function, &launch_cooperative_grid, &launch_pdl, &global_scratch_obj, &profile_scratch_obj,
                                           &kernel_metadata, &launch_metadata,
                                           &launch_enter_hook, &launch_exit_hook, &_arg0, &_arg1, &_arg2, &_arg3, &_arg4, &_arg5, &_arg6, &_arg7, &_arg8, &_arg9, &_arg10, &_arg11, &_arg12, &_arg13, &_arg14, &_arg15, &_arg16, &_arg17, &_arg18, &_arg19, &_arg20, &_arg21)) {
    return NULL;
  }

  int num_warps, num_ctas, shared_memory, clusterDimX, clusterDimY, clusterDimZ;
  if (!PyArg_ParseTuple(kernel_metadata, "iiiiii", &num_warps, &num_ctas, &shared_memory, &clusterDimX, &clusterDimY, &clusterDimZ)) {
    PyErr_SetString(PyExc_TypeError, "kernel_metadata must be a tuple");
    return NULL;
  }

  // extract launch metadata
  if (launch_enter_hook != Py_None){
    PyObject* ret = PyObject_CallOneArg(launch_enter_hook, launch_metadata);
    if (!ret)
      return NULL;
    Py_DECREF(ret);
  }

  CUdeviceptr global_scratch = 0;
  if (global_scratch_obj != Py_None) {
    DevicePtrInfo global_scratch_info = getPointer(global_scratch_obj, -1);
    if (!global_scratch_info.valid) {
      return NULL;
    }
    global_scratch = global_scratch_info.dev_ptr;
  }

  CUdeviceptr profile_scratch = 0;
  if (profile_scratch_obj != Py_None) {
    DevicePtrInfo profile_scratch_info = getPointer(profile_scratch_obj, -1);
    if (!profile_scratch_info.valid) {
      return NULL;
    }
    profile_scratch = profile_scratch_info.dev_ptr;
  }

  // raise exception asap
  
  CUtensorMap* tma_ptr0 = getTmaDesc(_arg0); if (!tma_ptr0) return NULL;
  CUtensorMap* tma_ptr5 = getTmaDesc(_arg5); if (!tma_ptr5) return NULL;
  CUtensorMap* tma_ptr10 = getTmaDesc(_arg10); if (!tma_ptr10) return NULL;
  
  Py_BEGIN_ALLOW_THREADS;
  _launch(gridX, gridY, gridZ, num_warps, num_ctas, launch_cooperative_grid, launch_pdl, clusterDimX, clusterDimY, clusterDimZ, shared_memory, (CUstream)_stream, (CUfunction)_function, global_scratch, profile_scratch, *tma_ptr0, _arg1, _arg2, _arg3, _arg4, *tma_ptr5, _arg6, _arg7, _arg8, _arg9, *tma_ptr10, _arg11, _arg12, _arg13, _arg14, _arg15, _arg16, _arg17);
  Py_END_ALLOW_THREADS;
  if (PyErr_Occurred()) {
    return NULL;
  }

  if(launch_exit_hook != Py_None){
    PyObject* ret = PyObject_CallOneArg(launch_exit_hook, launch_metadata);
    if (!ret)
      return NULL;
    Py_DECREF(ret);
  }

  Py_RETURN_NONE;
}

static PyMethodDef ModuleMethods[] = {
  {"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"},
  {NULL, NULL, 0, NULL} // sentinel
};

static struct PyModuleDef ModuleDef = {
  PyModuleDef_HEAD_INIT,
  "__triton_launcher",
  NULL, //documentation
  -1, //size
  ModuleMethods
};

PyMODINIT_FUNC PyInit___triton_launcher(void) {
  data_ptr_str = PyUnicode_InternFromString("data_ptr");
  if(data_ptr_str == NULL) {
    return NULL;
  }
  PyObject* driver_mod = PyImport_ImportModule("triton.backends.nvidia.driver");
  if (driver_mod == NULL) {
    return NULL;
  }
  py_tensor_map_type = PyObject_GetAttrString(driver_mod, "PyCUtensorMap");
  if (py_tensor_map_type == NULL) {
    return NULL;
  }

  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {
    return NULL;
  }
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}
