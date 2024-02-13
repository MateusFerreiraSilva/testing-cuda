#include <stdio.h>

/*
 * print a grid with 4 block each with a 8 X 8 matrix
*/
__global__ void print_thread_ids() {
    printf("thread_id_x: %d, thread_id_y: %d, thread_id_z: %d\n", threadIdx.x, threadIdx.y, threadIdx.z);
}

int main() {
    int nx = 16, ny = 16;

    dim3 block(8, 8);
    dim3 grid(nx / block.x, ny / block.y);

    print_thread_ids<<<grid, block>>>();

    cudaDeviceSynchronize();


    return 0;
}