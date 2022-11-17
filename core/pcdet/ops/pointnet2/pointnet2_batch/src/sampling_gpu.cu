/*
batch version of point sampling and gathering, modified from the original implementation of official PointNet++ codes.
Written by Shaoshuai Shi
All Rights Reserved 2018.
*/


#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"
#include "sampling_gpu.h"
#include <thrust/device_ptr.h>


__global__ void gather_points_kernel_fast(int b, int c, int n, int m, 
    const float *__restrict__ points, const int *__restrict__ idx, float *__restrict__ out) {
    // points: (B, C, N)
    // idx: (B, M)
    // output:
    //      out: (B, C, M)

    int bs_idx = blockIdx.z;
    int c_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b || c_idx >= c || pt_idx >= m) return;

    out += bs_idx * c * m + c_idx * m + pt_idx;
    idx += bs_idx * m + pt_idx;
    points += bs_idx * c * n + c_idx * n;
    out[0] = points[idx[0]];
}

void gather_points_kernel_launcher_fast(int b, int c, int n, int npoints, 
    const float *points, const int *idx, float *out) {
    // points: (B, C, N)
    // idx: (B, npoints)
    // output:
    //      out: (B, C, npoints)

    cudaError_t err;
    dim3 blocks(DIVUP(npoints, THREADS_PER_BLOCK), c, b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    gather_points_kernel_fast<<<blocks, threads>>>(b, c, n, npoints, points, idx, out);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

__global__ void gather_points_grad_kernel_fast(int b, int c, int n, int m, const float *__restrict__ grad_out, 
    const int *__restrict__ idx, float *__restrict__ grad_points) {
    // grad_out: (B, C, M)
    // idx: (B, M)
    // output:
    //      grad_points: (B, C, N)

    int bs_idx = blockIdx.z;
    int c_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b || c_idx >= c || pt_idx >= m) return;

    grad_out += bs_idx * c * m + c_idx * m + pt_idx;
    idx += bs_idx * m + pt_idx;
    grad_points += bs_idx * c * n + c_idx * n;

    atomicAdd(grad_points + idx[0], grad_out[0]);
}

void gather_points_grad_kernel_launcher_fast(int b, int c, int n, int npoints, 
    const float *grad_out, const int *idx, float *grad_points) {
    // grad_out: (B, C, npoints)
    // idx: (B, npoints)
    // output:
    //      grad_points: (B, C, N)

    cudaError_t err;
    dim3 blocks(DIVUP(npoints, THREADS_PER_BLOCK), c, b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    gather_points_grad_kernel_fast<<<blocks, threads>>>(b, c, n, npoints, grad_out, idx, grad_points);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}


__device__ void __update(float *__restrict__ dists, int *__restrict__ dists_i, int idx1, int idx2){
    const float v1 = dists[idx1], v2 = dists[idx2];
    const int i1 = dists_i[idx1], i2 = dists_i[idx2];
    dists[idx1] = max(v1, v2);
    dists_i[idx1] = v2 > v1 ? i2 : i1;
}

template <unsigned int block_size>
__global__ void farthest_point_sampling_kernel(int b, int n, int m,
    const float *__restrict__ dataset, float *__restrict__ temp, int *__restrict__ idxs) {
    // dataset: (B, N, 3)
    // tmp: (B, N)
    // output:
    //      idx: (B, M)

    if (m <= 0) return;
    __shared__ float dists[block_size]; // 读取block内的共享内存速度很快
    __shared__ int dists_i[block_size];

    int batch_index = blockIdx.x;
    dataset += batch_index * n * 3;   // 第i个batch的输入点
    temp += batch_index * n;  // float的距离，用来保存每个输入点云到当前采样点集合的最近距离
    idxs += batch_index * m;  // 第i个batch的输出索引内容

    int tid = threadIdx.x;  // 第j个线程
    const int stride = block_size; // block尺寸(1-1024)，和输入点数的2次幂有关，当输入点对于1024个后stride恒为1024

    int old = 0;  // 上一个采样点
    if (threadIdx.x == 0)  // 第1个线程，设置初始采样点为第0个输入点
    idxs[0] = old;

    __syncthreads();  // 同步block(batch)内的所有线程
    for (int j = 1; j < m; j++) {  // 第2个到最后一个采样点
    int besti = 0;
    float best = -1;
    float x1 = dataset[old * 3 + 0];  // 上一个采样点坐标
    float y1 = dataset[old * 3 + 1];
    float z1 = dataset[old * 3 + 2];
    // 上次采样后，采样点集合新增1个点，因此需要更新每个输入点到采样点集合的最近距离temp(点i到集合中每个点的最近距离)。
    // 只需要计算第j个输入点云之前到采样点集合的最近距离和上一个采样点的距离比较大小。
    for (int k = tid; k < n; k += stride) {  // 第k个线程，负责(i,i+stride,i+stridex2,...,k)，即把输入点数分成s个1024维
        float x2, y2, z2; // 第k = i(线程id) + n x striede 的坐标
        x2 = dataset[k * 3 + 0];
        y2 = dataset[k * 3 + 1];
        z2 = dataset[k * 3 + 2];
        // float mag = (x2 * x2) + (y2 * y2) + (z2 * z2);
        // if (mag <= 1e-3)
        // continue;

        float d = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);
        float d2 = min(d, temp[k]); // 点云k到上一个采样点的距离比到其他采样点的距离小。更新输入点k到采样点的最小距离。
        temp[k] = d2;
        besti = d2 > best ? k : besti; //  选取到当前采样点集合最远的线程对应的n个区间输入点云(输入点云到采样点集合最小值距离最大的那个)。
        best = d2 > best ? d2 : best;
    }
    dists[tid] = best;
    dists_i[tid] = besti;
    __syncthreads(); // 同步，所有线程，因为这里需要等到所有线程把dist和dist_i全部填写完毕。

    // 下面的方法是等分区间逐步并行取最大值：划分多个区间，每个在其内部区间内并行比较大小。然后最终得到最大值。
    // 比如1-16个数取最大值，则1-8和8-16先比(结果放在1-8),然后1-4和4-8比...
    if (block_size >= 1024) {  // 前一半线程和对应的后一半线程比较，选取最大的距离和索引更新到前一半线程位姿上。
        if (tid < 512) {
            __update(dists, dists_i, tid, tid + 512);
        }
        __syncthreads();  // 等待前半个线程都与后半个线程完成比较。
    }

    if (block_size >= 512) { // 前四分之一...
        if (tid < 256) {
            __update(dists, dists_i, tid, tid + 256);
        }
        __syncthreads();
    }
    if (block_size >= 256) { // 前八分之一...
        if (tid < 128) {
            __update(dists, dists_i, tid, tid + 128);
        }
        __syncthreads();
    }
    if (block_size >= 128) {
        if (tid < 64) {
            __update(dists, dists_i, tid, tid + 64);
        }
        __syncthreads();
    }
    if (block_size >= 64) {
        if (tid < 32) {
            __update(dists, dists_i, tid, tid + 32);
        }
        __syncthreads();
    }
    if (block_size >= 32) {
        if (tid < 16) {
            __update(dists, dists_i, tid, tid + 16);
        }
        __syncthreads();
    }
    if (block_size >= 16) {
        if (tid < 8) {
            __update(dists, dists_i, tid, tid + 8);
        }
        __syncthreads();
    }
    if (block_size >= 8) {
        if (tid < 4) {
            __update(dists, dists_i, tid, tid + 4);
        }
        __syncthreads();
    }
    if (block_size >= 4) {
        if (tid < 2) {
            __update(dists, dists_i, tid, tid + 2);
        }
        __syncthreads();
    }
    if (block_size >= 2) {
        if (tid < 1) {
            __update(dists, dists_i, tid, tid + 1);
        }
        __syncthreads();
    }

    // 最大距离(所有输入点里，当前采样点集合最近距离最大的点)对应的索引为本次采样点
    old = dists_i[0]; // 更新最近一次的采样点为本次采样点，用于更新输入点云到采样点集合的最小值。
    if (tid == 0)
        idxs[j] = old; // 某个线程写入输出即可，否则多个线程会并行写。
    }
}

void farthest_point_sampling_kernel_launcher(int b, int n, int m,
    const float *dataset, float *temp, int *idxs) {
    // dataset: (B, N, 3)
    // tmp: (B, N)
    // output:
    //      idx: (B, M)

    cudaError_t err;
    unsigned int n_threads = opt_n_threads(n);

    switch (n_threads) {
        case 1024:
        farthest_point_sampling_kernel<1024><<<b, n_threads>>>(b, n, m, dataset, temp, idxs); break;
        case 512:
        farthest_point_sampling_kernel<512><<<b, n_threads>>>(b, n, m, dataset, temp, idxs); break;
        case 256:
        farthest_point_sampling_kernel<256><<<b, n_threads>>>(b, n, m, dataset, temp, idxs); break;
        case 128:
        farthest_point_sampling_kernel<128><<<b, n_threads>>>(b, n, m, dataset, temp, idxs); break;
        case 64:
        farthest_point_sampling_kernel<64><<<b, n_threads>>>(b, n, m, dataset, temp, idxs); break;
        case 32:
        farthest_point_sampling_kernel<32><<<b, n_threads>>>(b, n, m, dataset, temp, idxs); break;
        case 16:
        farthest_point_sampling_kernel<16><<<b, n_threads>>>(b, n, m, dataset, temp, idxs); break;
        case 8:
        farthest_point_sampling_kernel<8><<<b, n_threads>>>(b, n, m, dataset, temp, idxs); break;
        case 4:
        farthest_point_sampling_kernel<4><<<b, n_threads>>>(b, n, m, dataset, temp, idxs); break;
        case 2:
        farthest_point_sampling_kernel<2><<<b, n_threads>>>(b, n, m, dataset, temp, idxs); break;
        case 1:
        farthest_point_sampling_kernel<1><<<b, n_threads>>>(b, n, m, dataset, temp, idxs); break;
        default:
        farthest_point_sampling_kernel<512><<<b, n_threads>>>(b, n, m, dataset, temp, idxs);
    }

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

template <unsigned int block_size>
__global__ void furthest_point_sampling_matrix_kernel(int b, int n, int m,
    const float *__restrict__ matrix, float *__restrict__ temp, int *__restrict__ idxs) {
    // distance_matrix: (B, N, N)
    // tmp: (B, N)
    // output:
    //      idx: (B, M)

    if (m <= 0) return;
    __shared__ float dists[block_size];
    __shared__ int dists_i[block_size];

    int batch_index = blockIdx.x;
    matrix += batch_index * n * n;
    temp += batch_index * n;
    idxs += batch_index * m;

    int tid = threadIdx.x;
    const int stride = block_size;

    int old = 0;
    if (threadIdx.x == 0)
    idxs[0] = old;

    __syncthreads();
    for (int j = 1; j < m; j++) {
    int besti = 0;
    float best = -1;
    for (int k = tid; k < n; k += stride) {
        float d = matrix[old * n + k];  // matrix[old][k]
        float d2 = min(d, temp[k]);
        temp[k] = d2;
        besti = d2 > best ? k : besti;
        best = d2 > best ? d2 : best;
    }
    dists[tid] = best;
    dists_i[tid] = besti;
    __syncthreads();

    if (block_size >= 1024) {
        if (tid < 512) {
            __update(dists, dists_i, tid, tid + 512);
        }
        __syncthreads();
    }

    if (block_size >= 512) {
        if (tid < 256) {
            __update(dists, dists_i, tid, tid + 256);
        }
        __syncthreads();
    }
    if (block_size >= 256) {
        if (tid < 128) {
            __update(dists, dists_i, tid, tid + 128);
        }
        __syncthreads();
    }
    if (block_size >= 128) {
        if (tid < 64) {
            __update(dists, dists_i, tid, tid + 64);
        }
        __syncthreads();
    }
    if (block_size >= 64) {
        if (tid < 32) {
            __update(dists, dists_i, tid, tid + 32);
        }
        __syncthreads();
    }
    if (block_size >= 32) {
        if (tid < 16) {
            __update(dists, dists_i, tid, tid + 16);
        }
        __syncthreads();
    }
    if (block_size >= 16) {
        if (tid < 8) {
            __update(dists, dists_i, tid, tid + 8);
        }
        __syncthreads();
    }
    if (block_size >= 8) {
        if (tid < 4) {
            __update(dists, dists_i, tid, tid + 4);
        }
        __syncthreads();
    }
    if (block_size >= 4) {
        if (tid < 2) {
            __update(dists, dists_i, tid, tid + 2);
        }
        __syncthreads();
    }
    if (block_size >= 2) {
        if (tid < 1) {
            __update(dists, dists_i, tid, tid + 1);
        }
        __syncthreads();
    }

    old = dists_i[0];
    if (tid == 0)
        idxs[j] = old;
    }
}

void furthest_point_sampling_matrix_kernel_launcher(int b, int n, int m,
    const float *matrix, float *temp, int *idxs) {
    // distance_matrix: (B, N, N)
    // tmp: (B, N)
    // output:
    //      idx: (B, M)

    cudaError_t err;
    unsigned int n_threads = opt_n_threads(n);

    switch (n_threads) {
        case 1024:
        furthest_point_sampling_matrix_kernel<1024><<<b, n_threads>>>(b, n, m, matrix, temp, idxs); break;
        case 512:
        furthest_point_sampling_matrix_kernel<512><<<b, n_threads>>>(b, n, m, matrix, temp, idxs); break;
        case 256:
        furthest_point_sampling_matrix_kernel<256><<<b, n_threads>>>(b, n, m, matrix, temp, idxs); break;
        case 128:
        furthest_point_sampling_matrix_kernel<128><<<b, n_threads>>>(b, n, m, matrix, temp, idxs); break;
        case 64:
        furthest_point_sampling_matrix_kernel<64><<<b, n_threads>>>(b, n, m, matrix, temp, idxs); break;
        case 32:
        furthest_point_sampling_matrix_kernel<32><<<b, n_threads>>>(b, n, m, matrix, temp, idxs); break;
        case 16:
        furthest_point_sampling_matrix_kernel<16><<<b, n_threads>>>(b, n, m, matrix, temp, idxs); break;
        case 8:
        furthest_point_sampling_matrix_kernel<8><<<b, n_threads>>>(b, n, m, matrix, temp, idxs); break;
        case 4:
        furthest_point_sampling_matrix_kernel<4><<<b, n_threads>>>(b, n, m, matrix, temp, idxs); break;
        case 2:
        furthest_point_sampling_matrix_kernel<2><<<b, n_threads>>>(b, n, m, matrix, temp, idxs); break;
        case 1:
        furthest_point_sampling_matrix_kernel<1><<<b, n_threads>>>(b, n, m, matrix, temp, idxs); break;
        default:
        furthest_point_sampling_matrix_kernel<512><<<b, n_threads>>>(b, n, m, matrix, temp, idxs);
    }

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

template <unsigned int block_size>
__global__ void furthest_point_sampling_weights_kernel(int b, int n, int m,
    const float *__restrict__ xyz, const float *__restrict__ weights, float *__restrict__ temp, int *__restrict__ idxs) {
    // xyz: (B, N, 3)
    // weights: (B, N)
    // tmp: (B, N)
    // output:
    //      idx: (B, M)

    if (m <= 0) return;
    __shared__ float dists[block_size];
    __shared__ int dists_i[block_size];

    int batch_index = blockIdx.x;
    xyz += batch_index * n * 3;
    weights += batch_index * n;
    temp += batch_index * n;
    idxs += batch_index * m;

    int tid = threadIdx.x;
    const int stride = block_size;

    int old = 0;

    __syncthreads();
    for (int j = 0; j < m; j++) {
    int besti = 0;
    float best = -1;
    float x1 = xyz[old * 3 + 0];
    float y1 = xyz[old * 3 + 1];
    float z1 = xyz[old * 3 + 2];
    for (int k = tid; k < n; k += stride) {
        if (j == 0) {  // select the point with the largest weight in the first round
            float d = weights[k];
            besti = d > best ? k : besti;
            best = d > best ? d : best;
        }
        else {
            float x2, y2, z2;
            x2 = xyz[k * 3 + 0];
            y2 = xyz[k * 3 + 1];
            z2 = xyz[k * 3 + 2];

            float d = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);
            d = min(d, temp[k]);
            temp[k] = d;
            float d2 = d * max(weights[k], 1e-12);  // dist[old][k] * weights[k]
            besti = d2 > best ? k : besti;
            best = d2 > best ? d2 : best;
        }
    }
    dists[tid] = best;
    dists_i[tid] = besti;
    __syncthreads();

    if (block_size >= 1024) {
        if (tid < 512) {
            __update(dists, dists_i, tid, tid + 512);
        }
        __syncthreads();
    }

    if (block_size >= 512) {
        if (tid < 256) {
            __update(dists, dists_i, tid, tid + 256);
        }
        __syncthreads();
    }
    if (block_size >= 256) {
        if (tid < 128) {
            __update(dists, dists_i, tid, tid + 128);
        }
        __syncthreads();
    }
    if (block_size >= 128) {
        if (tid < 64) {
            __update(dists, dists_i, tid, tid + 64);
        }
        __syncthreads();
    }
    if (block_size >= 64) {
        if (tid < 32) {
            __update(dists, dists_i, tid, tid + 32);
        }
        __syncthreads();
    }
    if (block_size >= 32) {
        if (tid < 16) {
            __update(dists, dists_i, tid, tid + 16);
        }
        __syncthreads();
    }
    if (block_size >= 16) {
        if (tid < 8) {
            __update(dists, dists_i, tid, tid + 8);
        }
        __syncthreads();
    }
    if (block_size >= 8) {
        if (tid < 4) {
            __update(dists, dists_i, tid, tid + 4);
        }
        __syncthreads();
    }
    if (block_size >= 4) {
        if (tid < 2) {
            __update(dists, dists_i, tid, tid + 2);
        }
        __syncthreads();
    }
    if (block_size >= 2) {
        if (tid < 1) {
            __update(dists, dists_i, tid, tid + 1);
        }
        __syncthreads();
    }

    old = dists_i[0];
    if (tid == 0)
        idxs[j] = old;
    }
}

void furthest_point_sampling_weights_kernel_launcher(int b, int n, int m,
    const float *xyz, const float *weights, float *temp, int *idxs) {
    // xyz: (B, N, 3)
    // weights: (B, N)
    // tmp: (B, N)
    // output:
    //      idx: (B, M)

    cudaError_t err;
    unsigned int n_threads = opt_n_threads(n);

    switch (n_threads) {
        case 1024:
        furthest_point_sampling_weights_kernel<1024><<<b, n_threads>>>(b, n, m, xyz, weights, temp, idxs); break;
        case 512:
        furthest_point_sampling_weights_kernel<512><<<b, n_threads>>>(b, n, m, xyz, weights, temp, idxs); break;
        case 256:
        furthest_point_sampling_weights_kernel<256><<<b, n_threads>>>(b, n, m, xyz, weights, temp, idxs); break;
        case 128:
        furthest_point_sampling_weights_kernel<128><<<b, n_threads>>>(b, n, m, xyz, weights, temp, idxs); break;
        case 64:
        furthest_point_sampling_weights_kernel<64><<<b, n_threads>>>(b, n, m, xyz, weights, temp, idxs); break;
        case 32:
        furthest_point_sampling_weights_kernel<32><<<b, n_threads>>>(b, n, m, xyz, weights, temp, idxs); break;
        case 16:
        furthest_point_sampling_weights_kernel<16><<<b, n_threads>>>(b, n, m, xyz, weights, temp, idxs); break;
        case 8:
        furthest_point_sampling_weights_kernel<8><<<b, n_threads>>>(b, n, m, xyz, weights, temp, idxs); break;
        case 4:
        furthest_point_sampling_weights_kernel<4><<<b, n_threads>>>(b, n, m, xyz, weights, temp, idxs); break;
        case 2:
        furthest_point_sampling_weights_kernel<2><<<b, n_threads>>>(b, n, m, xyz, weights, temp, idxs); break;
        case 1:
        furthest_point_sampling_weights_kernel<1><<<b, n_threads>>>(b, n, m, xyz, weights, temp, idxs); break;
        default:
        furthest_point_sampling_weights_kernel<512><<<b, n_threads>>>(b, n, m, xyz, weights, temp, idxs);
    }

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
