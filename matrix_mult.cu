#include <stdio.h>

__global__ void matrix_mult(float *A, float *B, float *C, const int N) {
    const int col_offset = blockIdx.x * blockDim.x;
    const int col = threadIdx.x + col_offset;
    const int row_offset = blockIdx.y * blockDim.y;
    const int row = threadIdx.y + row_offset;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += A[row * N + i] * B[i * N + col];
        }

        C[row * N + col] = sum;
    }
}

// TO DO check the existence condition of the matrix
int main() {
    const int BLOCK_SIZE = 32;

    const int N = 2;
    const int SIZE = N * N;
    const int SIZE_IN_BYTES = SIZE * sizeof(float);

    // host data

    /*
     * flattened matrix:
     * [0.5 1]
     * [0.2 0.8]
    */
    float h_A[] = { 0.5, 1, 0.2, 0.8}; // allocated on stack

    /*
     * flattened matrix:
     * [0.92 0.4]
     * [1 0.15]
    */
    float h_B[] = { 0.92, 0.4, 1, 0.15 }; // allocated on stack
    float *h_C = (float*)malloc(SIZE_IN_BYTES);


    // device data
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, SIZE_IN_BYTES);
    cudaMalloc((void**)&d_B, SIZE_IN_BYTES);
    cudaMalloc((void**)&d_C, SIZE_IN_BYTES);

    // memory transfer from host to device
    cudaMemcpy(d_A, h_A, SIZE_IN_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, SIZE_IN_BYTES, cudaMemcpyHostToDevice);

    dim3 block(N, N);
    dim3 grid(1);

    matrix_mult<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, SIZE_IN_BYTES, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    for (int i = 0; i < SIZE; i++) {
        printf("\t%2.2f", h_C[i]);
        if ((i + 1) % N == 0) {
            printf("\n");
        }
    }

    free(h_C);

    return 0;
}