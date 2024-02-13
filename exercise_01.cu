#include <stdio.h>

__global__ void print_grid_and_block_info() {
    printf("thread_id_x: %d, block_id_x: %d, grid_dim_x: %d, grid_dim_y: %d, grid_dim_z: %d\n", threadIdx.x, blockIdx.y, gridDim.x, gridDim.y, gridDim.z);
}

int main () {

    dim3 block(2, 2, 2);
    dim3 grid(4, 4, 4);

    print_grid_and_block_info<<<grid, block>>>();

    cudaDeviceSynchronize();

    return 0;
}