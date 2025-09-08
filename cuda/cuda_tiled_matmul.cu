// Tiled matrix multiplication, block size 8x8
// each block calculates one tile of the result matrix

#include <stdio.h>

#define TILE_WIDTH 2
#define N 4 // assuming square matrices of size NxN for simplicity

__global__ void tiledMatrixMultiplication(float *A, float *B, float *C) {
    // tile_A spans rows of A, tile_B spans columns of B over iterations
    __shared__ float tile_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile_B[TILE_WIDTH][TILE_WIDTH];

    float element_value = 0;
    for (int iteration = 0; iteration < N / TILE_WIDTH; iteration++) {
                                          // ROW                              COL                         each element of the tile  
        tile_A[threadIdx.y][threadIdx.x] = A[N * TILE_WIDTH * blockIdx.y    + TILE_WIDTH * iteration    + N * threadIdx.y + threadIdx.x];
        tile_B[threadIdx.y][threadIdx.x] = B[N * TILE_WIDTH * iteration     + TILE_WIDTH * blockIdx.x   + N * threadIdx.y + threadIdx.x];
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++) {
            element_value += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }
        __syncthreads();
    }
    C[N * TILE_WIDTH * blockIdx.y + TILE_WIDTH * blockIdx.x + N * threadIdx.y + threadIdx.x] = element_value;
}

int main() {
    float h_A[N][N], h_B[N][N], h_C[N][N];
    float *d_A, *d_B, *d_C;

    // Initialize matrices A and B with some values
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            h_A[i][j] = static_cast<float>(1.0f);
            h_B[i][j] = static_cast<float>(i - j);
        }
    }

    // Allocate device memory
    cudaMalloc((void**)&d_A, N * N * sizeof(float));
    cudaMalloc((void**)&d_B, N * N * sizeof(float));
    cudaMalloc((void**)&d_C, N * N * sizeof(float));

    // Copy matrices A and B to device
    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Define block and thread dimensions
    dim3 block = dim3(N / TILE_WIDTH, N / TILE_WIDTH);
    dim3 thread = dim3(TILE_WIDTH, TILE_WIDTH);

    // Launch kernel
    tiledMatrixMultiplication<<<block, thread>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();
    // Copy result matrix C back to host
    cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print result matrix C
    printf("Result matrix C:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%6.1f ", h_C[i][j]);
        }
        printf("\n");
    }

    return EXIT_SUCCESS;
}