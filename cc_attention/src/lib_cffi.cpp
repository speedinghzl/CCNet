// All functions assume that input and output tensors are already initialized
// and have the correct dimensions
#include <THC/THC.h>
#include <torch/extension.h>
#include "ca.h"

extern THCState *state;

int ca_forward_cuda(const at::Tensor& t, const at::Tensor& f, at::Tensor& weight) {
  cudaStream_t stream = THCState_getCurrentStream(state);
  int N, C, H, W;
  N = t.size(0); C = t.size(1); H = t.size(2); W = t.size(3);
  float * t_data = t.data<float>();
  float * f_data = f.data<float>();
  float * weight_data = weight.data<float>();
  return _ca_forward_cuda(N, C, H, W, t_data, f_data, weight_data, stream);
}

int ca_backward_cuda(const at::Tensor& dw, const at::Tensor& t, const at::Tensor& f, at::Tensor& dt, at::Tensor& df) {

  cudaStream_t stream = THCState_getCurrentStream(state);
  int N, C, H, W;
  N = t.size(0); C = t.size(1); H = t.size(2); W = t.size(3);
  float * t_data = t.data<float>();
  float * f_data = f.data<float>();
  float * dt_data = dt.data<float>();
  float * df_data = df.data<float>();
  float * dw_data = dw.data<float>();
  return _ca_backward_cuda(N, C, H, W, dw_data, t_data, f_data, dt_data, df_data, stream);
}

int ca_map_forward_cuda(const at::Tensor& weight, const at::Tensor& g, at::Tensor& out) {
  cudaStream_t stream = THCState_getCurrentStream(state);

  int N, C, H, W;
  N = g.size(0); C = g.size(1); H = g.size(2); W = g.size(3);

  const float *weight_data = weight.data<float>();
  const float *g_data = g.data<float>();
  float *out_data = out.data<float>();

  return _ca_map_forward_cuda(N, C, H, W, weight_data, g_data, out_data, stream);
}

int ca_map_backward_cuda(const at::Tensor& dout, const at::Tensor& weight, const at::Tensor& g,
                     at::Tensor& dw, at::Tensor& dg) {
  cudaStream_t stream = THCState_getCurrentStream(state);

  int N, C, H, W;
  N = dout.size(0); C = dout.size(1); H = dout.size(2); W = dout.size(3);

  const float *dout_data = dout.data<float>();
  const float *weight_data = weight.data<float>();
  const float *g_data = g.data<float>();
  float *dw_data = dw.data<float>();
  float *dg_data = dg.data<float>();

  return _ca_map_backward_cuda(N, C, H, W, dout_data, weight_data, g_data, dw_data, dg_data, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("ca_forward_cuda", &ca_forward_cuda, "CA forward CUDA");
    m.def("ca_backward_cuda", &ca_backward_cuda, "CA backward CUDA");
    m.def("ca_map_forward_cuda", &ca_map_forward_cuda, "CA map forward CUDA");
    m.def("ca_map_backward_cuda", &ca_map_backward_cuda, "CA map backward CUDA");
}

