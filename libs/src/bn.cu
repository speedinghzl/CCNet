#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>

#include "common.h"
#include "bn.h"

/*
 * Device functions and data structures
 */
struct Float2 {
  float v1, v2;
  __device__ Float2() {}
  __device__ Float2(float _v1, float _v2) : v1(_v1), v2(_v2) {}
  __device__ Float2(float v) : v1(v), v2(v) {}
  __device__ Float2(int v) : v1(v), v2(v) {}
  __device__ Float2 &operator+=(const Float2 &a) {
    v1 += a.v1;
    v2 += a.v2;
    return *this;
  }
};

struct SumOp {
  __device__ SumOp(const float *t, int c, int s)
      : tensor(t), C(c), S(s) {}
  __device__ __forceinline__ float operator()(int batch, int plane, int n) {
    return tensor[(batch * C + plane) * S + n];
  }
  const float *tensor;
  const int C;
  const int S;
};

struct VarOp {
  __device__ VarOp(float m, const float *t, int c, int s)
      : mean(m), tensor(t), C(c), S(s) {}
  __device__ __forceinline__ float operator()(int batch, int plane, int n) {
    float val = tensor[(batch * C + plane) * S + n];
    return (val - mean) * (val - mean);
  }
  const float mean;
  const float *tensor;
  const int C;
  const int S;
};

struct GradOp {
  __device__ GradOp(float _gamma, float _beta, const float *_z, const float *_dz, int c, int s)
      : gamma(_gamma), beta(_beta), z(_z), dz(_dz), C(c), S(s) {}
  __device__ __forceinline__ Float2 operator()(int batch, int plane, int n) {
    float _y = (z[(batch * C + plane) * S + n] - beta) / gamma;
    float _dz = dz[(batch * C + plane) * S + n];
    return Float2(_dz, _y * _dz);
  }
  const float gamma;
  const float beta;
  const float *z;
  const float *dz;
  const int C;
  const int S;
};

static __device__ __forceinline__ float warpSum(float val) {
#if __CUDA_ARCH__ >= 300
  for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
    val += WARP_SHFL_XOR(val, 1 << i, WARP_SIZE);
  }
#else
  __shared__ float values[MAX_BLOCK_SIZE];
  values[threadIdx.x] = val;
  __threadfence_block();
  const int base = (threadIdx.x / WARP_SIZE) * WARP_SIZE;
  for (int i = 1; i < WARP_SIZE; i++) {
    val += values[base + ((i + threadIdx.x) % WARP_SIZE)];
  }
#endif
  return val;
}

static __device__ __forceinline__ Float2 warpSum(Float2 value) {
  value.v1 = warpSum(value.v1);
  value.v2 = warpSum(value.v2);
  return value;
}

template <typename T, typename Op>
__device__ T reduce(Op op, int plane, int N, int C, int S) {
  T sum = (T)0;
  for (int batch = 0; batch < N; ++batch) {
    for (int x = threadIdx.x; x < S; x += blockDim.x) {
      sum += op(batch, plane, x);
    }
  }

  // sum over NumThreads within a warp
  sum = warpSum(sum);

  // 'transpose', and reduce within warp again
  __shared__ T shared[32];
  __syncthreads();
  if (threadIdx.x % WARP_SIZE == 0) {
    shared[threadIdx.x / WARP_SIZE] = sum;
  }
  if (threadIdx.x >= blockDim.x / WARP_SIZE && threadIdx.x < WARP_SIZE) {
    // zero out the other entries in shared
    shared[threadIdx.x] = (T)0;
  }
  __syncthreads();
  if (threadIdx.x / WARP_SIZE == 0) {
    sum = warpSum(shared[threadIdx.x]);
    if (threadIdx.x == 0) {
      shared[0] = sum;
    }
  }
  __syncthreads();

  // Everyone picks it up, should be broadcast into the whole gradInput
  return shared[0];
}

/*
 * Kernels
 */
__global__ void mean_var_kernel(const float *x, float *mean, float *var, int N,
                                int C, int S) {
  int plane = blockIdx.x;
  float norm = 1.f / (N * S);

  float _mean = reduce<float, SumOp>(SumOp(x, C, S), plane, N, C, S) * norm;
  __syncthreads();
  float _var = reduce<float, VarOp>(VarOp(_mean, x, C, S), plane, N, C, S) * norm;

  if (threadIdx.x == 0) {
    mean[plane] = _mean;
    var[plane] = _var;
  }
}

__global__ void forward_kernel(const float *x, const float *mean,
                               const float *var, const float *weight,
                               const float *bias, float *y, float *z, float eps,
                               int N, int C, int S) {
  int plane = blockIdx.x;

  float _mean = mean[plane];
  float _var = var[plane];
  float invStd = 0;
  if (_var != 0.f || eps != 0.f) {
    invStd = 1 / sqrt(_var + eps);
  }

  float gamma = weight != 0 ? abs(weight[plane]) + eps : 1.f;
  float beta = bias != 0 ? bias[plane] : 0.f;
  for (int batch = 0; batch < N; ++batch) {
    for (int n = threadIdx.x; n < S; n += blockDim.x) {
      float _x = x[(batch * C + plane) * S + n];
      float _y = (_x - _mean) * invStd;
      float _z = _y * gamma + beta;

      y[(batch * C + plane) * S + n] = _y;
      z[(batch * C + plane) * S + n] = _z;
    }
  }
}

__global__ void edz_eydz_kernel(const float *z, const float *dz, const float *weight, const float *bias,
                                float *edz, float *eydz, float eps, int N, int C, int S) {
  int plane = blockIdx.x;
  float norm = 1.f / (N * S);
  
  float gamma = weight != 0 ? abs(weight[plane]) + eps : 1.f;
  float beta = bias != 0 ? bias[plane] : 0.f;

  Float2 res = reduce<Float2, GradOp>(GradOp(gamma, beta, z, dz, C, S), plane, N, C, S);
  float _edz = res.v1 * norm;
  float _eydz = res.v2 * norm;
  __syncthreads();

  if (threadIdx.x == 0) {
    edz[plane] = _edz;
    eydz[plane] = _eydz;
  }
}

__global__ void backward_kernel(const float *dz, const float *z, const float *var, const float *weight,
                                const float *bias, const float *edz, const float *eydz, float *dx, float *dweight,
                                float *dbias, float eps, int N, int C, int S) {
  int plane = blockIdx.x;
  float _edz = edz[plane];
  float _eydz = eydz[plane];

  float gamma = weight != 0 ? abs(weight[plane]) + eps : 1.f;
  float beta = bias != 0 ? bias[plane] : 0.f;

  if (dx != 0) {
    float _var = var[plane];
    float invStd = 0;
    if (_var != 0.f || eps != 0.f) {
      invStd = 1 / sqrt(_var + eps);
    }

    float mul = gamma * invStd;

    for (int batch = 0; batch < N; ++batch) {
      for (int n = threadIdx.x; n < S; n += blockDim.x) {
        float _dz = dz[(batch * C + plane) * S + n];
        float _y = (z[(batch * C + plane) * S + n] - beta) / gamma;
        dx[(batch * C + plane) * S + n] = (_dz - _edz - _y * _eydz) * mul;
      }
    }
  }
  
  if (dweight != 0 || dbias != 0) {
    float norm = N * S;

    if (dweight != 0) {
      if (threadIdx.x == 0) {
        if (weight[plane] > 0)
          dweight[plane] += _eydz * norm;
        else if (weight[plane] < 0)
          dweight[plane] -= _eydz * norm;
      }
    }

    if (dbias != 0) {
      if (threadIdx.x == 0) {
        dbias[plane] += _edz * norm;
      }
    }
  }
}

/*
 * Implementations
 */
extern "C" int _bn_mean_var_cuda(int N, int C, int S, const float *x, float *mean,
                                 float *var, cudaStream_t stream) {
  // Run kernel
  dim3 blocks(C);
  dim3 threads(getNumThreads(S));
  mean_var_kernel<<<blocks, threads, 0, stream>>>(x, mean, var, N, C, S);

  // Check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    return 0;
  else
    return 1;
}

extern "C" int _bn_forward_cuda(int N, int C, int S, const float *x,
                                const float *mean, const float *var,
                                const float *weight, const float *bias, float *y,
                                float *z, float eps, cudaStream_t stream) {
  // Run kernel
  dim3 blocks(C);
  dim3 threads(getNumThreads(S));
  forward_kernel<<<blocks, threads, 0, stream>>>(x, mean, var, weight, bias, y,
                                                 z, eps, N, C, S);

  // Check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    return 0;
  else
    return 1;
}

extern "C" int _bn_edz_eydz_cuda(int N, int C, int S, const float *z, const float *dz, const float *weight,
                                    const float *bias, float *edz, float *eydz, float eps, cudaStream_t stream) {
  // Run kernel
  dim3 blocks(C);
  dim3 threads(getNumThreads(S));
  edz_eydz_kernel<<<blocks, threads, 0, stream>>>(z, dz, weight, bias, edz, eydz, eps, N, C, S);

  // Check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    return 0;
  else
    return 1;
}

extern "C" int _bn_backward_cuda(int N, int C, int S, const float *dz, const float *z, const float *var,
                                    const float *weight, const float *bias, const float *edz, const float *eydz,
                                    float *dx, float *dweight, float *dbias, float eps, cudaStream_t stream) {
  // Run kernel
  dim3 blocks(C);
  dim3 threads(getNumThreads(S));
  backward_kernel<<<blocks, threads, 0, stream>>>(dz, z, var, weight, bias, edz, eydz, dx, dweight, dbias,
                                                  eps, N, C, S);

  // Check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    return 0;
  else
    return 1;
}

extern "C" int _leaky_relu_cuda(int N, float *x, float slope, cudaStream_t stream) {
  // Run using thrust
  thrust::device_ptr<float> th_x = thrust::device_pointer_cast(x);
  thrust::transform_if(thrust::cuda::par.on(stream), th_x, th_x + N, th_x,
                       [slope] __device__ (const float& x) { return x * slope; },
                       [] __device__ (const float& x) { return x < 0; });

  // Check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    return 0;
  else
    return 1;
}

extern "C" int _leaky_relu_backward_cuda(int N, const float *x, float *dx, float slope, cudaStream_t stream) {
  // Run using thrust
  thrust::device_ptr<const float> th_x = thrust::device_pointer_cast(x);
  thrust::device_ptr<float> th_dx = thrust::device_pointer_cast(dx);
  thrust::transform_if(thrust::cuda::par.on(stream), th_dx, th_dx + N, th_x, th_dx,
                       [slope] __device__ (const float& dx) { return dx * slope; },
                       [] __device__ (const float& x) { return x < 0; });

  // Check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    return 0;
  else
    return 1;
}

extern "C" int _elu_cuda(int N, float *x, cudaStream_t stream) {
  // Run using thrust
  thrust::device_ptr<float> th_x = thrust::device_pointer_cast(x);
  thrust::transform_if(thrust::cuda::par.on(stream), th_x, th_x + N, th_x,
                       [] __device__ (const float& x) { return exp(x) - 1.f; },
                       [] __device__ (const float& x) { return x < 0; });

  // Check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    return 0;
  else
    return 1;
}

extern "C" int _elu_backward_cuda(int N, const float *x, float *dx, cudaStream_t stream) {
  // Run using thrust
  thrust::device_ptr<const float> th_x = thrust::device_pointer_cast(x);
  thrust::device_ptr<float> th_dx = thrust::device_pointer_cast(dx);
  thrust::transform_if(thrust::cuda::par.on(stream), th_dx, th_dx + N, th_x, th_x, th_dx,
                       [] __device__ (const float& dx, const float& x) { return dx * (x + 1.f); },
                       [] __device__ (const float& x) { return x < 0; });

  // Check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    return 0;
  else
    return 1;
}

extern "C" int _elu_inv_cuda(int N, float *x, cudaStream_t stream) {
  // Run using thrust
  thrust::device_ptr<float> th_x = thrust::device_pointer_cast(x);
  thrust::transform_if(thrust::cuda::par.on(stream), th_x, th_x + N, th_x,
                       [] __device__ (const float& x) { return log1p(x); },
                       [] __device__ (const float& x) { return x < 0; });

  // Check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    return 0;
  else
    return 1;
}
