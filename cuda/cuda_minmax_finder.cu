#include <stdio.h>
#include <cfloat>

template <typename T> __device__ T getMin(const T &a, const T &b) {
    return (a < b) ? a : b;
}
template <typename T> __device__ T getMax(const T &a, const T &b) {
    return (a > b) ? a : b;
}

template <typename T>
__global__ void reduceArray(const T *input, T *blockMin, T *blockMax, int size) {
    extern __shared__ T sharedMemory[];
    T *sharedMin = sharedMemory;
    T *sharedMax = sharedMemory + blockDim.x;

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    // Initialize shared memory
    sharedMin[tid] = (idx < size) ? input[idx] : FLT_MAX;
    sharedMax[tid] = (idx < size) ? input[idx] : -FLT_MAX;
    __syncthreads();

    // Intra-block reduction
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sharedMin[tid] = getMin(sharedMin[tid], sharedMin[tid + stride]);
            sharedMax[tid] = getMax(sharedMax[tid], sharedMax[tid + stride]);
        }
        __syncthreads();
    }

    // Write block result to global memory
    if (tid == 0) {
        blockMin[blockIdx.x] = sharedMin[0];
        blockMax[blockIdx.x] = sharedMax[0];
    }
}

// AI generated kernel -- does same thing as reduceArray but for final reduction at block level.
// TODO: Refactor such that we don't need two separate kernels.
// have output of first kernel be "block level min/max", then launch the kernel again with that output as input.
// This does work, but its kinda inefficient. 
template <typename T>
__global__ void finalReduce(const T *blockMin, const T *blockMax, T *minOutput, T *maxOutput, int numBlocks) {
    extern __shared__ T sharedMemory[];
    T *sharedMin = sharedMemory;
    T *sharedMax = sharedMemory + blockDim.x;

    int tid = threadIdx.x;

    sharedMin[tid] = (tid < numBlocks) ? blockMin[tid] : FLT_MAX;
    sharedMax[tid] = (tid < numBlocks) ? blockMax[tid] : -FLT_MAX;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride && tid + stride < numBlocks) {
            sharedMin[tid] = getMin(sharedMin[tid], sharedMin[tid + stride]);
            sharedMax[tid] = getMax(sharedMax[tid], sharedMax[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        *minOutput = sharedMin[0];
        *maxOutput = sharedMax[0];
    }
}

template <typename T>
void findMinMax(const T *input, T *minOutput, T *maxOutput, int size) {
    T *d_input, *d_blockMin, *d_blockMax, *d_minOutput, *d_maxOutput;
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    cudaMalloc(&d_input, size * sizeof(T));
    cudaMalloc(&d_blockMin, numBlocks * sizeof(T));
    cudaMalloc(&d_blockMax, numBlocks * sizeof(T));
    cudaMalloc(&d_minOutput, sizeof(T));
    cudaMalloc(&d_maxOutput, sizeof(T));

    cudaMemcpy(d_input, input, size * sizeof(T), cudaMemcpyHostToDevice);

    reduceArray<<<numBlocks, blockSize, blockSize * sizeof(T) * 2>>>(d_input, d_blockMin, d_blockMax, size);
    finalReduce<<<1, blockSize, blockSize * sizeof(T) * 2>>>(d_blockMin, d_blockMax, d_minOutput, d_maxOutput, numBlocks);

    cudaMemcpy(minOutput, d_minOutput, sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(maxOutput, d_maxOutput, sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_blockMin);
    cudaFree(d_blockMax);
    cudaFree(d_minOutput);
    cudaFree(d_maxOutput);
}

int main() {
    const int arraySize = 1024;
    float h_input[arraySize];
    for (int i = 0; i < arraySize; ++i) {
        float random_value = static_cast<float>(rand()) / RAND_MAX;
        h_input[i] = random_value * 100.0f; // Random values between 0 and 100
        printf("%f ", h_input[i]);
    }
    printf("\n");

    float h_min, h_max;
    findMinMax(h_input, &h_min, &h_max, arraySize);

    printf("\n Min: %f, Max: %f\n", h_min, h_max);
    return 0;
}
