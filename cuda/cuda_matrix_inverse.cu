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

// Each thread normalizes an element in a row by dividing its corresponding element by the pivot element.
__global__ void reduceRow(float* AB, int pivot_row, float pivot_value, int N, int M) {
    int col = threadIdx.x;
    if (col < N + M) {
        AB[pivot_row * (N + M) + col] /= pivot_value;
    }
}

// Each thread eliminates a row, by subtracting pivot row multiplied by the appropriate factor.
__global__ void eliminateRow(float* AB, int pivot_row, int N, int M, int i) {
    int row = threadIdx.x;
    if (row != pivot_row && row < N) {
        float factor = AB[row * (N + M) + i];
        for (int col = 0; col < N + M; col++) {
            AB[row * (N + M) + col] -= factor * AB[pivot_row * (N + M) + col];
        }
    }
}


int main() {
    // User changes matrix size and values here
    int N = 3;
    int M = 3;
    float h_A[9] = {1, 1, 0, 1, 1, 1, 1, 0, 1}; // Example 3x3 matrix
    float h_B[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1}; // Identity matrix
    // end User changeable section

    float *d_A, *d_B, *d_AB;
    cudaMalloc((void**)&d_A, N * N * sizeof(int));
    cudaMalloc((void**)&d_B, N * M * sizeof(int));
    cudaMemcpy(d_A, h_A, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * M * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_AB, N * (N + M) * sizeof(int));

    joinMatrix<<<1, N>>>(d_A, d_B, d_AB, N, M);
    cudaDeviceSynchronize();

    float h_AB[18];

    // Find pivot for ith column, and reduce and eliminate rows based on the pivot.
    // j is the row of the pivot, searched from i to N.
    // if pivot is not at (i, i), swap rows i and j.
    int j;
    for (int i = 0; i < N; i++) {
        j = i;
        cudaMemcpy(h_AB, d_AB, N * (N + M) * sizeof(int), cudaMemcpyDeviceToHost);
        
        float pivot_value = h_AB[j * (N + M) + i];
        while (pivot_value == 0) {
            j++;
            if (j >= N) {
                printf("Matrix is singular and cannot be inverted.\n");
                return EXIT_FAILURE;
            }
            pivot_value = h_AB[j * (N + M) + i];
        }

        reduceRow<<<1, N + M>>>(d_AB, j, pivot_value, N, M); // Normalize pivot row
        cudaDeviceSynchronize();
        eliminateRow<<<1, N>>>(d_AB, j, N, M, i); // Eliminate other rows
        cudaDeviceSynchronize();

        if (j != i) {
            // Swap rows in host memory
            cudaMemcpy(h_AB, d_AB, N * (N + M) * sizeof(int), cudaMemcpyDeviceToHost);
            for (int col = 0; col < N + M; col++) {
                float temp = h_AB[i * (N + M) + col];
                h_AB[i * (N + M) + col] = h_AB[j * (N + M) + col];
                h_AB[j * (N + M) + col] = temp;
            }
            cudaMemcpy(d_AB, h_AB, N * (N + M) * sizeof(int), cudaMemcpyHostToDevice);
        }
    }

    cudaMemcpy(h_AB, d_AB, N * (N + M) * sizeof(int), cudaMemcpyDeviceToHost);

    float *h_solution = new float[N * M];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            h_solution[i * M + j] = h_AB[i * (N + M) + (N + j)];
        }
    }
    
    printf("Solution Matrix:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            printf("%f ", h_solution[i * M + j]);
        }
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_AB);
    delete[] h_solution;
    
    return EXIT_SUCCESS;
}
