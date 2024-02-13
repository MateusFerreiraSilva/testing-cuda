#include <stdio.h>

__global__ void sum_array_gpu(int *a, int *b, int *result, int length) {
    const int offset = blockIdx.x * blockDim.x;
    const int gid = threadIdx.x + offset;

    if (gid < length) {
        result[gid] = a[gid] + b[gid];
    }
}

void sum_array_cpu(int *a, int *b, int *result, int length) {
    for (int i = 0; i < length; i++) {
        result[i] = a[i] + b[i];
    }
}

void compare_arrays(int *cpu_result, int *gpu_result, int length) {
    for (int i = 0; i < length; i++) {
        if (cpu_result[i] != gpu_result[i]) {
            printf("Arrays are different\n");

            return;
        }
    }

    printf("Arrays are the same\n");
}

int main() {
    const int length = 10000;
    const int block_size = 128;

    const int byte_size = length * sizeof(int);

    // host pointers
    int *h_a = (int*) malloc(byte_size);
    int *h_b = (int*) malloc(byte_size);
    int *h_result_gpu = (int*) malloc(byte_size);
    int *h_result_cpu = (int*) malloc(byte_size);

    // initalizing host pointers
    srand(time(NULL));
    for (int i = 0; i < length; i++) {
        h_a[i] = rand() % 100 + 1;
    }

    for (int i = 0; i < length; i++) {
        h_b[i] = rand() % 100 + 1;
    }

    sum_array_cpu(h_a, h_b, h_result_cpu, length);

    // device pointers
    int *d_a, *d_b, *d_result;
    cudaMalloc((int**)&d_a, byte_size);
    cudaMalloc((int**)&d_b, byte_size);
    cudaMalloc((int**)&d_result, byte_size);

    // memory transfer from host to device
    cudaMemcpy(d_a, h_a, byte_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, byte_size, cudaMemcpyHostToDevice);

    dim3 block(block_size);
    dim3 grid(length / block.x + 1);

    sum_array_gpu<<<grid, block>>>(d_a, d_b, d_result, length);
    cudaDeviceSynchronize();

    cudaMemcpy(h_result_gpu, d_result, byte_size, cudaMemcpyDeviceToHost);

    free(h_a);
    free(h_b);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);

    compare_arrays(h_result_cpu, h_result_gpu, length);

    free(h_result_gpu);
    free(h_result_cpu);

    return 0;
}