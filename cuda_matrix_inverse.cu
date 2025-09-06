// CUDA accelerated matrix inversion using Gauss-Jordan elimination
// some parts adapted from:
// https://www.sciencedirect.com/science/article/pii/S0045794913002095

#include <stdio.h>

// Joins two matrices A (N x N) and B (N x M) into a single (N x (N+M)) matrix AB
// Each thread handles a row of the resulting matrix.
__global__ void joinMatrix(float* A, float* B, float* AB, int N, int M) {
    int row = threadIdx.x;
    if (row < N) {
        for (int col = 0; col < N; col++) {
            AB[row * (N + M) + col] = A[row * N + col];
        }
        for (int col = 0; col < M; col++) {
            AB[row * (N + M) + (N + col)] = B[row * M + col];
        }
    }
}

int main() {
    int N = 3;
    int M = 3;
    float h_A[9] = {4, 7, 2, 3, 6, 1, 2, 5, 3}; // Example 3x3 matrix
    float h_B[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1}; // Identity matrix

    float *d_A, *d_B, *d_AB;
    cudaMalloc((void**)&d_A, N * N * sizeof(int));
    cudaMalloc((void**)&d_B, N * M * sizeof(int));
    cudaMemcpy(d_A, h_A, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * M * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_AB, N * (N + M) * sizeof(int));

    joinMatrix<<<1, N>>>(d_A, d_B, d_AB, N, M);
    cudaDeviceSynchronize();

    float h_AB[18];
    cudaMemcpy(h_AB, d_AB, N * (N + M) * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Joined Matrix AB:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N + M; j++) {
            printf("%f ", h_AB[i * (N + M) + j]);
        }
        printf("\n");
    }

    
}
