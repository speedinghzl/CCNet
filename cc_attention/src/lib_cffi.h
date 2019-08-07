int ca_forward_cuda(const at::Tensor& t, const at::Tensor& *f, at::Tensor& weight);
int ca_backward_cuda(const at::Tensor& dw, const at::Tensor& t, const at::Tensor& f, at::Tensor& dt, at::Tensor& df);

int ca_map_forward_cuda(const at::Tensor& weight, const at::Tensor& g, at::Tensor& out);
int ca_map_backward_cuda(const at::Tensor& dout, const at::Tensor& weight, const at::Tensor& g, at::Tensor& dw, at::Tensor& dg);
