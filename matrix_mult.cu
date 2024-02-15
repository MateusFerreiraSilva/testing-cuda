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

int main()
{
    const int BLOCK_SIZE = 32;
    dim3 A_dim, B_dim;

    printf("Input the dimensions (x, y) of matrix A: ");
    scanf("%d%d", &A_dim.x, &A_dim.y);

    printf("Input the dimensions (x, y) of matrix B: ");
    scanf("%d%d", &B_dim.x, &B_dim.y);

    if (A_dim.y != B_dim.x) { // matrix existence condition
        printf("Invalid dimensions. The number of columns in Matrix A must be equal to the number of rows in Matrix B.\n");

        return 0;
    }

    const int N = A_dim.y; // or B_dim.x
    const dim3 C_dim(A_dim.x, B_dim.y);

    thrust::host_vector<float> h_A(A_dim.x * A_dim.y); // flattened matrix A
    printf("Input matrix A:\n");
    for (int i = 0; i < A_dim.x; i++) {
        for (int j = 0; j < A_dim.y; j++) {
            scanf("%f", &h_A[i * A_dim.x + j]);
        }
    }

    thrust::host_vector<float> h_B(B_dim.x * B_dim.y); // flattened matrix B
    printf("Input matrix B:\n");
    for (int i = 0; i < B_dim.x; i++) {
        for (int j = 0; j < B_dim.y; j++) {
            scanf("%f", &h_B[i * B_dim.x + j]);
        }
    }

    thrust::device_vector<float> d_A = h_A;
    thrust::device_vector<float> d_B = h_B;
    thrust::device_vector<float> d_C(C_dim.x  * C_dim.y);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(1);

    matrix_mult<<<grid, block>>>(
        thrust::raw_pointer_cast(d_A.data()), thrust::raw_pointer_cast(d_B.data()),
        thrust::raw_pointer_cast(d_C.data()), N);
    cudaDeviceSynchronize();

    thrust::host_vector<float> h_C = d_C;

    printf("\n");
    for (int i = 0; i < h_C.size(); i++)
    {
        printf("\t%2.2f", h_C[i]);
        if ((i + 1) % C_dim.y == 0) // this means new row
        {
            printf("\n");
        }
    }

    return 0;
}