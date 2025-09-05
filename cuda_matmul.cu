#include <stdio.h>

// Calculates the Matric multiplication of first * second, stores into result
// first matrix has dimensions N x M
// second matrix has dimensions M x L
// result matrix has dimensions N x L
__global__ void MatMul(int first[], int second[], int result[], int N, int M, int L) {
    int row = threadIdx.x;
    int col = threadIdx.y;
    
    int sum = 0;
    for (int k = 0; k < M; k++) {
        sum += first[row * M + k] * second[k * L + col];
    }

    result[row * L + col] = sum;
    printf("Row: %d, Col: %d, Value: %d\n", row, col, sum);
}

int main() {
    // USER SHOULD ONLY CHANGE THIS PART
    int A_rows = 3, A_cols = 2;
    int B_rows = 2, B_cols = 2;
    int* A = new int[A_rows * A_cols] {1, 2, 3, 4, 5, 6};
    int* B = new int[B_rows * B_cols] {1, 2, 3, 4};
    // END OF USER CHANGEABLE PART

    if (A_cols != B_rows) {
        printf("Incompatible matrix dimensions for multiplication.\n");
        return EXIT_FAILURE;
    }

    int *d_A, *d_B, *result;
    cudaMalloc((void**)&d_A,    A_rows * A_cols * sizeof(int));
    cudaMalloc((void**)&d_B,    B_rows * B_cols * sizeof(int));
    cudaMalloc((void**)&result, A_rows * B_cols * sizeof(int));
    

    cudaMemcpy(d_A, A, 3 * 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, 2 * 2 * sizeof(int), cudaMemcpyHostToDevice);

    dim3 numBlocks          = dim3(1, 1, 1);
    dim3 threadsPerBlock    = dim3(A_rows, B_cols, 1);

    MatMul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, result, A_rows, A_cols, B_cols);
    
    cudaDeviceSynchronize();

    int* resultHost = new int[A_rows * B_cols];
    cudaMemcpy(resultHost, result, A_rows * B_cols * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Result matrix:\n");
    for (int i = 0; i < A_rows; i++) {
        for (int j = 0; j < B_cols; j++) {
            printf("%d ", resultHost[i * 2 + j]);
        }
        printf("\n");
    }

    return EXIT_SUCCESS;
}