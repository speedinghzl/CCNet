#ifndef __BN__
#define __BN__

/*
 * Exported functions
 */
extern "C" int _bn_mean_var_cuda(int N, int C, int S, const float *x, float *mean, float *var, cudaStream_t);
extern "C" int _bn_forward_cuda(int N, int C, int S, const float *x, const float *mean, const float *var,
                                const float *weight, const float *bias, float *y, float *z, float eps, cudaStream_t);
extern "C" int _bn_edz_eydz_cuda(int N, int C, int S, const float *z, const float *dz, const float *weight,
                                 const float *bias, float *edz, float *eydz, float eps, cudaStream_t stream);
extern "C" int _bn_backward_cuda(int N, int C, int S, const float *dz, const float *z, const float *var,
                                 const float *weight, const float *bias, const float *edz, const float *eydz, float *dx,
                                 float *dweight, float *dbias, float eps, cudaStream_t stream);
extern "C" int _leaky_relu_cuda(int N, float *x, float slope, cudaStream_t stream);
extern "C" int _leaky_relu_backward_cuda(int N, const float *x, float *dx, float slope, cudaStream_t stream);
extern "C" int _elu_cuda(int N, float *x, cudaStream_t stream);
extern "C" int _elu_backward_cuda(int N, const float *x, float *dx, cudaStream_t stream);
extern "C" int _elu_inv_cuda(int N, float *x, cudaStream_t stream);

#endif
