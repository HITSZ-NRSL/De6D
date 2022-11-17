#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <curand_kernel.h>
#include <torch/serialize/tensor.h>
#include <vector>
#include <THC/THC.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include "gridify.h"


#define CHECK_CUDA(x) do { \
  if (!x.type().is_cuda()) { \
    fprintf(stderr, "%s must be CUDA tensor at %s:%d\n", #x, __FILE__, __LINE__); \
    exit(-1); \
  } \
} while (0)
#define CHECK_CONTIGUOUS(x) do { \
  if (!x.is_contiguous()) { \
    fprintf(stderr, "%s must be contiguous tensor at %s:%d\n", #x, __FILE__, __LINE__); \
    exit(-1); \
  } \
} while (0)
#define CHECK_MALLOC(x) do{int ret = x;printf("%s: %d\n",#x,ret);}while(0)

#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
#define TOTAL_THREADS 1024

#define ndim 3
#define data_ndim 4

// See: http://codeyarns.com/2011/03/02/how-to-do-error-checking-in-cuda/
inline void
__cuda_check_errors (const char *filename, const int line_number)
{
    cudaError err = cudaDeviceSynchronize ();
    if (err != cudaSuccess)
    {
        printf ("CUDA error %i at %s:%i: %s\n",
                err, filename, line_number, cudaGetErrorString (err));
        exit (-1);
    }
}

inline void
__cuda_safe_call (cudaError err, const char *filename, const int line_number)
{
    if (err != cudaSuccess)
    {
        printf ("CUDA error %i at %s:%i: %s\n",
                err, filename, line_number, cudaGetErrorString (err));
        exit (-1);
    }
}

template<typename Dtype>
__global__ void gridify_kernel_build_index(
        int *out_nebidx, Dtype *out_nebidxmsk, Dtype *out_cent,
        Dtype *out_centmsk, int *out_actual_centnum, int *actual_centcount,
        const Dtype *in_data, const int *in_actual_numpoints,
        const int B,
        const int N,
        const int max_o,
        const int P,
        const int kernel_size,
        const int stride,
        const int loc,
        const float *d_coord_shift,
        const float *d_voxel_size,
        const float *d_grid_size,
        const int grid_size_vol,
        const int size,
        int *coor_to_voxelidx,
        int *voxelidx_to_coor,
        int *coor_to_pntidx,
        float *coor_to_locxyzw,
        int *coor_counter
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x; // index of gpu thread
    int i_batch = index / N;  // index of batch
    if (i_batch >= B) { return; }
    int i_pt = index - N * i_batch;
//     printf("B = %d, i = %d, in_num = %d\n",i_batch,i_pt,in_actual_numpoints[i_batch]);
    if (i_pt < in_actual_numpoints[i_batch]) {
        curandState state;
        int coor[ndim];
        const Dtype *p_pt = in_data + index * data_ndim;
        for (int j = 0; j < ndim; j++) {
            int c = floor((p_pt[j] + d_coord_shift[j]) / d_voxel_size[j]);
            if (c < 0 || c >= d_grid_size[j]) {
                return;
            }
            coor[j] = c;
        }
        int coor_indx = coor[2] * (d_grid_size[0] * d_grid_size[1]) + coor[1] * d_grid_size[0] + coor[0];
        int coor_indx_b = i_batch * grid_size_vol + coor_indx;
        int grid_pntidx = atomicAdd(coor_counter + coor_indx_b, 1);
        if (grid_pntidx < P) {
//             printf("coord: {%d,%d,%d}, vid: %d, b_id: %d, p_id: %d,\n",coor[0],coor[1],coor[2],coor_indx_b,i_batch,i_pt);
            coor_to_pntidx[coor_indx_b * P + grid_pntidx] = i_pt;
        } else {
            curand_init(index, 0, 0, &state);
            int insrtidx = ceilf(curand_uniform(&state) * (grid_pntidx + 1)) - 1;
            if (insrtidx < P) {
                coor_to_pntidx[coor_indx_b * P + insrtidx] = i_pt;
            }
        }
        if (loc == 1) {
            int coor_b_idx = coor_indx_b * data_ndim;
            float weight = p_pt[3];
            atomicAdd(coor_to_locxyzw + coor_b_idx, p_pt[0] * weight);
            atomicAdd(coor_to_locxyzw + coor_b_idx + 1, p_pt[1] * weight);
            atomicAdd(coor_to_locxyzw + coor_b_idx + 2, p_pt[2] * weight);
            atomicAdd(coor_to_locxyzw + coor_b_idx + 3, weight);
        }


        int voxel_idx = coor_to_voxelidx[coor_indx_b];
        //        printf("grid_size_vol: %d, coor_index: %d, i_batch %d, voxel_idx: %d ; \n", grid_size_vol, coor_index, i_batch, voxel_idx);
        if (voxel_idx == -1) {  // found an empty voxel
            Dtype old_voxel_num = atomicCAS(
                    &coor_to_voxelidx[coor_indx_b],
                    -1, 0
            );
            if (old_voxel_num == -1) {
                // CAS -> old val, if old val is -1
                // if we get -1, this thread is the one who obtain a new voxel
                // so only this thread should do the increase operator below
                int tmp = atomicAdd(actual_centcount + i_batch, 1); // increase the counter, return old counter
                atomicAdd(out_actual_centnum + i_batch, 1); // increase the counter, return old counter
                if (tmp < max_o) {
                    voxelidx_to_coor[i_batch * max_o + tmp] = coor_indx;
                    out_centmsk[i_batch * max_o + tmp] = 1.0; // change center mask to 1 at new occupied voxel
                } else {
                    curand_init(index, 0, 0, &state);
                    int insrtidx = ceilf(curand_uniform(&state) * (tmp + 1)) - 1;
                    if (insrtidx < max_o) {
                        voxelidx_to_coor[i_batch * max_o + insrtidx] = coor_indx;
                    }
                }
                if (out_actual_centnum[i_batch] > max_o) {
                    out_actual_centnum[i_batch] = max_o;
                }
            }
        }
    }
}

template<typename Dtype>
__global__ void gridify_kernel_query_neighs(
        int *out_nebidx, Dtype *out_nebidxmsk, Dtype *out_cent, Dtype *out_centmsk, int *out_actual_centnum,
        const Dtype *in_data, const int *in_actual_numpoints,
        const int B,
        const int N,
        const int max_o,
        const int P,
        const int kernel_size,
        const int stride,
        const int loc,
        const float *d_coord_shift,
        const float *d_voxel_size,
        const float *d_grid_size,
        const int grid_size_vol,
        const int size,
        int *coor_to_voxelidx,
        int *voxelidx_to_coor,
        int *coor_to_pntidx,
        float *coor_to_locxyzw,
        int *coor_counter,
        int *voxelidx_counter
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x; // index of gpu thread
    int i_batch = index / max_o;  // index of batch
    if (i_batch >= B) { return; }
    int i_ogrid = index - i_batch * max_o;
    if (i_ogrid < out_actual_centnum[i_batch]) {
        curandState state;
        int coor_indx_b, in_data_eleweight, grid_pntidx = 0, insrtidx, idx, idx_b_data;
        float xsum = 0.0, ysum = 0.0, zsum = 0.0, countweightsum = 0, oldweight = 0;
        //        int coor_indx = coor[2] * (d_grid_size[0] * d_grid_size[1])
        //                        + coor[1] * d_grid_size[0] + coor[0];
        int coor = voxelidx_to_coor[index];
        int coor2 = coor / (d_grid_size[0] * d_grid_size[1]);
        int coor1 = (coor - coor2 * (d_grid_size[0] * d_grid_size[1])) / d_grid_size[0];
        int coor0 = coor - coor2 * (d_grid_size[0] * d_grid_size[1]) - coor1 * d_grid_size[0];
        int d_coor, h_coor, w_coor, initID;
        float total_weight = 0;
        int index_data = index * data_ndim;
        int index_P = index * P;
        int coor_indx_b_origin;
        for (int nei_idx = 0; nei_idx < size; nei_idx++) {
            d_coor = nei_idx / (kernel_size * kernel_size) - (kernel_size - 1) / 2 + coor2;
            h_coor = (nei_idx % (kernel_size * kernel_size)) / kernel_size - (kernel_size - 1) / 2 + coor1;
            w_coor = nei_idx % kernel_size - (kernel_size - 1) / 2 + coor0;
            if (d_coor >= 0 && d_coor < d_grid_size[2] && h_coor >= 0 &&
                h_coor < d_grid_size[1] && w_coor >= 0 && w_coor < d_grid_size[0]) {
                coor_indx_b = i_batch * grid_size_vol + d_coor * (d_grid_size[0] * d_grid_size[1])
                              + h_coor * d_grid_size[0] + w_coor;
                if (nei_idx * 2 + 1 == size) { coor_indx_b_origin = coor_indx_b; }
                int amount = min(P, coor_counter[coor_indx_b]);
                for (int j = 0; j < amount; j++) {
                    if (grid_pntidx++ < P) {
                        idx = coor_to_pntidx[coor_indx_b * P + j];
                        if (grid_pntidx == 1) { initID = idx; }
                        idx_b_data = (idx + N * i_batch) * data_ndim;
                        in_data_eleweight = in_data[idx_b_data + 3];
                        out_nebidx[index_P + grid_pntidx - 1] = idx;
                        out_nebidxmsk[index_P + grid_pntidx - 1] = 1.0;
                        total_weight += in_data_eleweight;
                    } else {
                        curand_init(index_P * size + grid_pntidx, 0, 0, &state);
                        insrtidx = ceilf(curand_uniform(&state) * (grid_pntidx)) - 1;
                        if (insrtidx < P) {
                            oldweight = in_data[(out_nebidx[index_P + insrtidx] + N * i_batch) * data_ndim + 3];
                            idx = coor_to_pntidx[coor_indx_b * P + j];
                            idx_b_data = (idx + N * i_batch) * data_ndim;
                            in_data_eleweight = in_data[idx_b_data + 3];
                            out_nebidx[index_P + insrtidx] = idx;
                            total_weight += (in_data_eleweight - oldweight);
                        }
                    }
                }
            }
        }
        out_cent[index_data + 3] = total_weight;
        if (grid_pntidx < P) {
            for (int j = grid_pntidx; j < P; j++) {
                out_nebidx[index_P + j] = initID;
            }
        }
        if (loc == 1) {
            int coor_indx_b_data = coor_indx_b_origin * data_ndim;
            xsum = coor_to_locxyzw[coor_indx_b_data];
            ysum = coor_to_locxyzw[coor_indx_b_data + 1];
            zsum = coor_to_locxyzw[coor_indx_b_data + 2];
            countweightsum = coor_to_locxyzw[coor_indx_b_data + 3];
            out_cent[index_data] = xsum / countweightsum;
            out_cent[index_data + 1] = ysum / countweightsum;
            out_cent[index_data + 2] = zsum / countweightsum;
        }
    }
}



int grid_query_wrapper_fast(
        at::Tensor nebidx,
        at::Tensor nebidxmsk,
        at::Tensor cent,
        at::Tensor centmsk,
        at::Tensor actual_centnum,
        at::Tensor data,
        at::Tensor actual_numpoints,
        std::vector<float> param_coord_shift,
        std::vector<float> param_grid_size,
        std::vector<float> param_voxel_size,
        std::vector<int> param_kernel_size
) {

    int stride = 1;
    int loc = 1;

    CHECK_INPUT(nebidx);
    CHECK_INPUT(nebidxmsk);
    CHECK_INPUT(cent);
    CHECK_INPUT(centmsk);
    CHECK_INPUT(actual_centnum);
    CHECK_INPUT(data);
    CHECK_INPUT(actual_numpoints);

    const int B = data.size(0);
    const int N = data.size(1);
    const int O = nebidx.size(1);
    const int P = nebidx.size(2);
    printf("B = %d; N = %d; M = %d; K = %d\n",B,N,O,P);

    int *out_nebidx = nebidx.data<int>();
    float *out_nebidxmsk = nebidxmsk.data<float>();
    float *out_cent = cent.data<float>();
    float *out_centmsk = centmsk.data<float>();
    int *out_actual_centnum = actual_centnum.data<int>();

    const float *in_data = data.data<float>();
    const int *in_actual_numpoints = actual_numpoints.data<int>();

    int grid_size_vol = (int) (param_grid_size[0] * param_grid_size[1] * param_grid_size[2]);
    const int size = param_kernel_size[0] * param_kernel_size[0] * param_kernel_size[0];
    printf("grid_vol = %d; kernel_vol = %d\n",grid_size_vol,size);

    float *coord_shift = new float[3];
    float *voxel_size = new float[3];
    float *grid_size = new float[3];
    for (int i = 0; i < 3; ++i) {
        coord_shift[i] = param_coord_shift[i];
        voxel_size[i] = param_voxel_size[i];
        grid_size[i] = param_grid_size[i];
    }
    printf("voxel: {%f,%f,%f}, grid: {%f,%f,%f}, shift: {%f,%f,%f}\n",
        voxel_size[0],voxel_size[1],voxel_size[2],
        grid_size[0],grid_size[1],grid_size[2],
        coord_shift[0],coord_shift[1],coord_shift[2]);

    float *d_coord_shift, *d_voxel_size, *d_grid_size;
    CHECK_MALLOC(cudaMalloc(&d_coord_shift, 3 * sizeof(float)));
    CHECK_MALLOC(cudaMalloc(&d_voxel_size, 3 * sizeof(float)));
    CHECK_MALLOC(cudaMalloc(&d_grid_size, 3 * sizeof(float)));
    cudaMemcpy(d_coord_shift, coord_shift, 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_voxel_size, voxel_size, 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grid_size, grid_size, 3 * sizeof(float), cudaMemcpyHostToDevice);

    float *d_coor_to_locxyzw;
    int *d_coor_to_voxelidx, *d_voxelidx_to_coor, *d_coor_to_pntidx, *d_coor_counter, *d_voxelidx_counter, *actual_centcount;
    // 这里的coor可能需要修改
    CHECK_MALLOC(cudaMalloc(&d_coor_to_locxyzw, B * grid_size_vol * 4 * sizeof(float)));
    CHECK_MALLOC(cudaMalloc(&d_coor_to_pntidx, B * grid_size_vol * P * sizeof(int)));
    CHECK_MALLOC(cudaMalloc(&d_coor_counter, B * grid_size_vol * sizeof(int)));
    CHECK_MALLOC(cudaMalloc(&d_coor_to_voxelidx, B * grid_size_vol * sizeof(int)));
    CHECK_MALLOC(cudaMalloc(&d_voxelidx_to_coor, B * O * sizeof(int)));
    CHECK_MALLOC(cudaMalloc(&d_voxelidx_counter, B * O * sizeof(int)));
    CHECK_MALLOC(cudaMalloc(&actual_centcount, B * sizeof(int)));
    cudaMemset(d_coor_to_locxyzw, 0, B * grid_size_vol * 4 * sizeof(float));
    cudaMemset(d_coor_counter, 0, B * grid_size_vol * sizeof(int));
    cudaMemset(d_voxelidx_counter, 0, B * O * sizeof(int));
    cudaMemset(d_coor_to_voxelidx, -1, B * grid_size_vol * sizeof(int));
    cudaMemset(actual_centcount, 0, B * sizeof(int));
    printf("coor_counter: %zu\n", B * grid_size_vol * sizeof(int));

    const int gridSize = (B * N + TOTAL_THREADS - 1) / TOTAL_THREADS;
    dim3 dimGrid(gridSize);
    dim3 dimBlock(TOTAL_THREADS);
    printf("build idx kernel launch: dimGrid = %d, dimBlock = %d \n", gridSize, TOTAL_THREADS);
    gridify_kernel_build_index<float><<<dimGrid, dimBlock>>>(
            out_nebidx, out_nebidxmsk, out_cent, out_centmsk, out_actual_centnum, actual_centcount, in_data, in_actual_numpoints,
                    B, N, O, P, param_kernel_size[0], stride, loc, d_coord_shift, d_voxel_size, d_grid_size,
                    grid_size_vol, size, d_coor_to_voxelidx, d_voxelidx_to_coor, d_coor_to_pntidx, d_coor_to_locxyzw, d_coor_counter);

    const int o_gridSize = (B * O + TOTAL_THREADS - 1) / TOTAL_THREADS;
    dim3 o_dimGrid(o_gridSize);
    dim3 o_dimBlock(TOTAL_THREADS);
    printf("query neighs kernel launch: dimGrid = %d, dimBlock = %d \n", o_gridSize, TOTAL_THREADS);
    gridify_kernel_query_neighs<float><<<o_dimGrid, o_dimBlock>>>(
            out_nebidx, out_nebidxmsk, out_cent, out_centmsk, out_actual_centnum, in_data, in_actual_numpoints,
                    B, N, O, P, param_kernel_size[0], stride, loc, d_coord_shift, d_voxel_size, d_grid_size,
                    grid_size_vol, size, d_coor_to_voxelidx, d_voxelidx_to_coor, d_coor_to_pntidx, d_coor_to_locxyzw,
                    d_coor_counter, d_voxelidx_counter);
//    MSHADOW_CUDA_POST_KERNEL_CHECK(cuda::gridify_kernel_query_neighs);
//     thrust::device_ptr<int> d_ptr=thrust::device_pointer_cast<int>(d_coor_to_voxelidx);
//     int LENGTH = B * grid_size_vol;
//     thrust::sort(d_ptr,d_ptr+LENGTH, thrust::greater<int>());
    printf("end kernel\n");
//     while(1){}
    delete coord_shift;
    delete voxel_size;
    delete grid_size;
    cudaFree(d_coord_shift);
    cudaFree(d_voxel_size);
    cudaFree(d_grid_size);
    cudaFree(d_coor_to_voxelidx);
    cudaFree(d_voxelidx_to_coor);
    cudaFree(d_coor_to_pntidx);
    cudaFree(d_coor_to_locxyzw);
    cudaFree(d_coor_counter);
    cudaFree(d_voxelidx_counter);
    cudaFree(actual_centcount);
    return 1;
}

