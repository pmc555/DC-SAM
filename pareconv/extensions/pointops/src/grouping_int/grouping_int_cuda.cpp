#include <torch/serialize/tensor.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>
//#include <THC/THC.h>

#include "grouping_int_cuda_kernel.h"

//extern THCState *state;

void grouping_int_forward_cuda(int b, int c, int n, int m, int nsample, at::Tensor points_tensor, at::Tensor idx_tensor, at::Tensor out_tensor)
{
    const int64_t *points = points_tensor.data_ptr<int64_t>();
    const int *idx = idx_tensor.data_ptr<int>();
    int64_t *out = out_tensor.data_ptr<int64_t>();
    grouping_int_forward_cuda_launcher(b, c, n, m, nsample, points, idx, out);
}

void grouping_int_forward_cuda_fast(int b, int c, int n, int m, int nsample, at::Tensor points_tensor, at::Tensor idx_tensor, at::Tensor out_tensor)
{
    const int64_t *points = points_tensor.data_ptr<int64_t>();
    const int *idx = idx_tensor.data_ptr<int>();
    int64_t *out = out_tensor.data_ptr<int64_t>();
    grouping_int_forward_cuda_launcher_fast(b, c, n, m, nsample, points, idx, out);
}