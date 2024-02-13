#include <stdio.h>

__global__ void mem_trs_test(int * input) {
    const int offset = blockIdx.x * blockDim.x;
    const int gid =  threadIdx.x + offset;
    printf("tid: %d, gid: %d, value : %d\n", threadIdx.x, gid, input[gid]);
}

int main() {
    const int length = 128;
    const int byte_size = length * sizeof(int);

    int *h_input = (int*) malloc(byte_size);

    for (int i = 0; i < length; i++) {
        h_input[i] = i + 1;
    }

    int *d_input;
    cudaMalloc((void**)&d_input, byte_size);

    cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice);

    dim3 block(64);
    dim3 grid(2);

    mem_trs_test<<<grid, block>>>(d_input);

    cudaDeviceSynchronize();

    free(h_input);
    cudaFree(d_input);

    return 0;
}