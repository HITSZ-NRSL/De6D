#ifndef _BALL_QUERY_GPU_H
#define _BALL_QUERY_GPU_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

int ball_query_wrapper_fast(int b, int n, int m, float radius, int nsample, 
	at::Tensor new_xyz_tensor, at::Tensor xyz_tensor, at::Tensor idx_tensor);

void ball_query_kernel_launcher_fast(int b, int n, int m, float radius, int nsample, 
	const float *xyz, const float *new_xyz, int *idx);

int ball_query_cnt_wrapper_fast(int b, int n, int m, float radius, int nsample,
	at::Tensor new_xyz_tensor, at::Tensor xyz_tensor, at::Tensor idx_cnt_tensor, at::Tensor idx_tensor);

void ball_query_cnt_kernel_launcher_fast(int b, int n, int m, float radius, int nsample,
	const float *xyz, const float *new_xyz, int *idx_cnt, int *idx);

int ball_query_dilated_wrapper_fast(int b, int n, int m, float radius_in, float radius_out, int nsample,
	at::Tensor new_xyz_tensor, at::Tensor xyz_tensor, at::Tensor idx_cnt_tensor, at::Tensor idx_tensor);

void ball_query_dilated_kernel_launcher_fast(int b, int n, int m, float radius_in, float radius_out, int nsample,
	const float *xyz, const float *new_xyz, int *idx_cnt, int *idx);


#endif
