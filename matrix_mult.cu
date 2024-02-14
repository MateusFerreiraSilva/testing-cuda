#include <stdio.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

__global__ void matrix_mult(
    float *A, float *B,
    float *C, const int N)
{
    const int col_offset = blockIdx.x * blockDim.x;
    const int col = threadIdx.x + col_offset;
    const int row_offset = blockIdx.y * blockDim.y;
    const int row = threadIdx.y + row_offset;

    if (row < N && col < N)
    {
        float sum = 0.0f;
        for (int i = 0; i < N; i++)
        {
            sum += A[row * N + i] * B[i * N + col];
        }

        C[row * N + col] = sum;
    }
}

// TO DO check the existence condition of the matrix
int main()
{
    const int BLOCK_SIZE = 32;

    const int N = 2;
    const int SIZE = N * N;

    // host data

    /*
     * flattened matrix:
     * [0.5 1]
     * [0.2 0.8]
     */

    thrust::host_vector<float> h_A = {0.5, 1, 0.2, 0.8}; // allocated on stack

    /*
     * flattened matrix:
     * [0.92 0.4]
     * [1 0.15]
     */
    thrust::host_vector<float> h_B = {0.92, 0.4, 1, 0.15}; // allocated on stack
    thrust::host_vector<float> h_C(SIZE);

    // device data
    thrust::device_vector<float> d_A = h_A;
    thrust::device_vector<float> d_B = h_B;
    thrust::device_vector<float> d_C(SIZE);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(1);

    matrix_mult<<<grid, block>>>(
        thrust::raw_pointer_cast(d_A.data()), thrust::raw_pointer_cast(d_B.data()),
        thrust::raw_pointer_cast(d_C.data()), N);
    cudaDeviceSynchronize();

    h_C = d_C;

    for (int i = 0; i < SIZE; i++)
    {
        printf("\t%2.2f", h_C[i]);
        if ((i + 1) % N == 0)
        {
            printf("\n");
        }
    }

    return 0;
}