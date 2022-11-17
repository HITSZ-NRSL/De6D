#include <torch/serialize/tensor.h>
#include <vector>
#include <THC/THC.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "gridify.h"

extern THCState *state;
//at::Tensor out_nebidx,
//at::Tensor out_nebidxmsk,
//at::Tensor out_cent,
//at::Tensor out_centmsk,
//at::Tensor out_actual_centnum,
//at::Tensor actual_centcount,
//at::Tensor in_data,
//at::Tensor in_actual_numpoints,
//const int B,
//const int N,
//const int max_o,
//const int P,
//const int kernel_size,
//const int stride,
//const int loc,
//at::Tensor d_coord_shift,
//at::Tensor d_voxel_size,
//at::Tensor d_grid_size,
//const int grid_size_vol,
//const int size,
//at::Tensor coor_to_voxelidx,
//at::Tensor voxelidx_to_coor,
//at::Tensor coor_to_pntidx,
//at::Tensor coor_to_locxyzw,
//at::Tensor coor_counter
