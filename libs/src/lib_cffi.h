int bn_mean_var_cuda(const THCudaTensor *x, THCudaTensor *mean, THCudaTensor *var);
int bn_forward_cuda(const THCudaTensor *x, const THCudaTensor *mean, const THCudaTensor *var,
                    const THCudaTensor *weight, const THCudaTensor *bias, THCudaTensor *y, THCudaTensor *z,
                    float eps);
int bn_edz_eydz_cuda(const THCudaTensor *z, const THCudaTensor *dz, const THCudaTensor *weight,
                     const THCudaTensor *bias, THCudaTensor *edz, THCudaTensor *eydz, float eps);
int bn_backard_cuda(const THCudaTensor *dz, const THCudaTensor *z, const THCudaTensor *var,
                    const THCudaTensor *weight, const THCudaTensor *bias, const THCudaTensor *edz,
                    const THCudaTensor *eydz, THCudaTensor *dx, THCudaTensor *dweight, THCudaTensor *dbias,
                    float eps);
int leaky_relu_cuda(THCudaTensor *x, float slope);
int leaky_relu_backward_cuda(const THCudaTensor *x, THCudaTensor *dx, float slope);
int elu_cuda(THCudaTensor *x);
int elu_backward_cuda(const THCudaTensor *x, THCudaTensor *dx);
int elu_inv_cuda(THCudaTensor *x);