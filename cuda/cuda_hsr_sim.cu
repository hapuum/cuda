// Parses Reliquary Archiver JSON file and fills in information for HSR relics and characters
// Reliquary repository found at: https://github.com/IceDynamix/reliquary-archiver/tree/main
// The parser itself is simple and mainly focuses on converting JSON file to structs to work with in C++.

// Process in 3 steps:
// 1. Parellel Scan JSON file to generate structural index arrays
// 2. Parellel parse JSON file based on structural index arrays to generate HSR Relic data
// 3. Parellel generate HSR Relic data to HSR Sim data

// includes and defines
#include <iostream>
#include <fstream>
#include <string>
#include "cuda_hsr_sim.h"

#define FILEPATH "/home/hapuum/cuda_learn/resource/relic_data.json"
#define TEST_FILEPATH "/home/hapuum/cuda_learn/resource/test.json"
#define CHAR_PER_THREAD 4096        // 1 full memory page
#define NUM_PER_BLOCK 256           // makes 1 block = 1MiB
#define T0_DEFAULT_MAX_SIZE 32768  // default size of DeviceVector at thread 0 vs rest.
#define REST_DEFAULT_MAX_SIZE 64  // thread 0 is reserved more space for reduction stage

template<typename T>
class
DeviceVector 
{
    T* data;
    size_t capacity, length;
    public:
        __device__ DeviceVector() : data(nullptr), capacity(16), length(0) 
        {
            data = new T[capacity];
        }
        __device__ DeviceVector(size_t c) : data(nullptr), capacity(c), length(0) 
        {
            data = new T[capacity];
        }

        __device__ ~DeviceVector() { delete[] data; }

        __device__ void push_back(const T& value) 
        {
            if (length >= capacity) 
            {
                // Expand capacity
                capacity *= 2;
                T* new_data = new T[capacity];
                for (size_t i = 0; i < length; ++i) new_data[i] = data[i];
                delete[] data;
                data = new_data;
            }
            data[length++] = value;
        }

        __device__ void set(size_t idx, const T& value) { data[idx] = value; }

        __device__ void join(DeviceVector<T>* next) 
        {
            for (int i = 0; i < next->length; i++) {
                this->push_back(next->get(i));
            }
        }

        __device__ T* getInternalArray() {
            return data;
        }

        __host__ __device__ void printVector() {
            for (int i = 0; i < this->length; i++) {
                printf("%d \n", this->get(i));
            }
            printf("\n");
        }

        // host can only read and cannot modify.
        __host__ __device__ T& operator[](size_t idx) { return data[idx]; }
        __host__ __device__ size_t size() const { return length; }
        __host__ __device__ T& get(size_t idx) { return data[idx]; }

};

__global__
void
build_structural_index
(
    // input
    char* file_data, 
    size_t file_size,
    // output
    int* open_brace_positions,
    int* close_brace_positions,
    int* open_bracket_positions,
    int* close_bracket_positions,
    int* colon_positions,
    int* comma_positions
) 
{
    // save index from last occurrence in a matrix, each row = each thread, index 0~5 in this order
    // later used for joining the vectors.
    __shared__ int last_occurrence[NUM_PER_BLOCK][6]; 

    int open_brace_last_occurrence = 0;
    int close_brace_last_occurrence = 0;
    int open_bracket_last_occurrence = 0;
    int close_bracket_last_occurrence = 0;
    int colon_last_occurrence = 0;
    int comma_last_occurrence = 0;

    size_t vector_capacity = (threadIdx.x == 0) ? T0_DEFAULT_MAX_SIZE : REST_DEFAULT_MAX_SIZE;

    DeviceVector<int>* open_brace_vector = new DeviceVector<int>(vector_capacity);
    DeviceVector<int>* close_brace_vector = new DeviceVector<int>(vector_capacity);
    DeviceVector<int>* open_bracket_vector = new DeviceVector<int>(vector_capacity);
    DeviceVector<int>* close_bracket_vector = new DeviceVector<int>(vector_capacity);
    DeviceVector<int>* colon_vector = new DeviceVector<int>(vector_capacity);
    DeviceVector<int>* comma_vector = new DeviceVector<int>(vector_capacity);

    // iterate through each character of assigned section of file_data
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = 0; i < CHAR_PER_THREAD; i++) {
        // update occurrence counter
        open_brace_last_occurrence++;
        close_brace_last_occurrence++;
        open_bracket_last_occurrence++;
        close_bracket_last_occurrence++;
        colon_last_occurrence++;
        comma_last_occurrence++;

        char c = file_data[idx * CHAR_PER_THREAD + i];
        switch (c) {
            case '{':
                open_brace_vector->push_back(open_brace_last_occurrence);
                open_brace_last_occurrence = 0;
                //printf("{ found at %d, thread: %d, block: %d, i: %d \n", idx * CHAR_PER_THREAD + i, threadIdx.x, blockIdx.x, i);
                break;
            case '}':
                close_brace_vector->push_back(close_brace_last_occurrence);
                close_brace_last_occurrence = 0;
                break;
            case '[':
                open_bracket_vector->push_back(open_bracket_last_occurrence);
                open_bracket_last_occurrence = 0;
                break;
            case ']':
                close_bracket_vector->push_back(close_bracket_last_occurrence);
                close_bracket_last_occurrence = 0;
                break;
            case ':':
                colon_vector->push_back(open_brace_last_occurrence);
                colon_last_occurrence = 0;
                break;
            case ',':
                comma_vector->push_back(open_brace_last_occurrence);
                comma_last_occurrence = 0;
                break;
        }
    }

    /*
    open_brace_vector->push_back(-1 * open_brace_last_occurrence);
    close_brace_vector->push_back(-1 * close_brace_last_occurrence);
    open_bracket_vector->push_back(-1 * open_bracket_last_occurrence);
    close_bracket_vector->push_back(-1 * close_bracket_last_occurrence);
    colon_vector->push_back(-1 * colon_last_occurrence);
    comma_vector->push_back(-1 * comma_last_occurrence);
    */

    last_occurrence[threadIdx.x][0] = -1 * open_brace_last_occurrence;
    last_occurrence[threadIdx.x][1] = -1 * close_brace_last_occurrence;
    last_occurrence[threadIdx.x][2] = -1 * open_bracket_last_occurrence;
    last_occurrence[threadIdx.x][3] = -1 * close_bracket_last_occurrence;
    last_occurrence[threadIdx.x][4] = -1 * colon_last_occurrence;
    last_occurrence[threadIdx.x][5] = -1 * comma_last_occurrence;
    
    __syncthreads();

    // DEBUG - thread level scanning correctness
    // if (threadIdx.x == 1) {
    //     open_brace_vector->printVector();
    //     printf("----------------------------------\n");
    // }

    // adjust value of "first occurrence" by carryover amount stored in last_occurrence matrix from previous thread
    if (threadIdx.x > 0) {
        open_brace_vector->     set(0, open_brace_vector      ->get(0)   - last_occurrence[threadIdx.x - 1][0]);
        close_brace_vector->    set(0, close_brace_vector     ->get(0)   - last_occurrence[threadIdx.x - 1][1]);
        open_bracket_vector->   set(0, open_bracket_vector    ->get(0)   - last_occurrence[threadIdx.x - 1][2]);
        close_bracket_vector->  set(0, close_bracket_vector   ->get(0)   - last_occurrence[threadIdx.x - 1][3]);
        colon_vector->          set(0, colon_vector           ->get(0)   - last_occurrence[threadIdx.x - 1][4]);
        comma_vector->          set(0, comma_vector           ->get(0)   - last_occurrence[threadIdx.x - 1][5]);
    }

    // DEBUG - Coalesce thread boundary values
    // if (threadIdx.x == 1) {
    //     printf("should be subtracted by %d \n", last_occurrence[0][0]);
    //     open_brace_vector->printVector();
    // }

    // BEHAVIOR UP TO THIS POINT SEEMS CORRECT

    __syncthreads();
    
    // join vectors by reduction
    // initialize 2D array of pointers to vectors storing each thread's vectors
    // might be able to optimize memory use by casting last_occurrence matrix to void* then to DeviceVector<int>*
    // basically halves the memory allocation, but should be pretty small overhead
    // compared to the actual reduction stage in double for loop below.
    // TODO: dynamically allocate vector_array and last_occurrence matrix instead of static allocation based on num_per_block

    __shared__ DeviceVector<int>* vector_array[NUM_PER_BLOCK][6];
    vector_array[threadIdx.x][0] = open_brace_vector;
    vector_array[threadIdx.x][1] = close_brace_vector;
    vector_array[threadIdx.x][2] = open_bracket_vector;
    vector_array[threadIdx.x][3] = close_bracket_vector;
    vector_array[threadIdx.x][4] = colon_vector;
    vector_array[threadIdx.x][5] = comma_vector;

    for (int step = 1; step < blockDim.x; step *= 2) {
        if (threadIdx.x % (2 * step) == 0) {
            for (int i = 0; i < 6; i++) {
                vector_array[threadIdx.x][i]->join(vector_array[threadIdx.x + step][i]);
            }
        }
        __syncthreads();
        // DEBUG - reduction
        // if (threadIdx.x == 0) {
        //     printf("thread 0, step %d, should cover 2 * %d threads \n", step, step);
        //     printf("--------------------------------------------\n");
        //     vector_array[threadIdx.x][0]->printVector();
        // }
    }
    
    // assign return values
    // work 6 threads for copying each array to output 
    switch (threadIdx.x) {
        case 0:
            for (int i = 0; i < vector_array[0][threadIdx.x]->size(); i++) {
                open_brace_positions[i] = vector_array[0][threadIdx.x]->get(i);
            }
            break;
        case 1:
            for (int i = 0; i < vector_array[0][threadIdx.x]->size(); i++) {
                close_brace_positions[i] = vector_array[0][threadIdx.x]->get(i);
            }
            break;
        case 2:
            for (int i = 0; i < vector_array[0][threadIdx.x]->size(); i++) {
                open_bracket_positions[i] = vector_array[0][threadIdx.x]->get(i);
            }
            break;
        case 3:
            for (int i = 0; i < vector_array[0][threadIdx.x]->size(); i++) {
                close_bracket_positions[i] = vector_array[0][threadIdx.x]->get(i);
            }
            break;
        case 4:
            for (int i = 0; i < vector_array[0][threadIdx.x]->size(); i++) {
                colon_positions[i] = vector_array[0][threadIdx.x]->get(i);
            }
            break;
        case 5:
            for (int i = 0; i < vector_array[0][threadIdx.x]->size(); i++) {
                comma_positions[i] = vector_array[0][threadIdx.x]->get(i);
            }
            break;
        default: 
            break;
    }

    if (threadIdx.x == 0) {
        // DEBUG - return val
        // vector_array[threadIdx.x][0]->printVector();
    }
    __syncthreads();


}

int main() {
    std::cout << "HSR Reliquary Archiver JSON Parser" << std::endl;
    // load json file
    bool testing = 0;
    std::string filepath = (testing) ? TEST_FILEPATH : FILEPATH;

    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filepath << std::endl;
        return EXIT_FAILURE;
    }
    std::string json_content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();
    std::cout << "File loaded successfully. Size: " << json_content.size() << " bytes" << std::endl;

    // scan json file to generate structural indices
    char* d_json_content;
    
    int* openBracePositions      = (int*)malloc(sizeof(int) * T0_DEFAULT_MAX_SIZE);
    int* closeBracePositions     = (int*)malloc(sizeof(int) * T0_DEFAULT_MAX_SIZE);
    int* openBracketPositions    = (int*)malloc(sizeof(int) * T0_DEFAULT_MAX_SIZE);
    int* closeBracketPositions   = (int*)malloc(sizeof(int) * T0_DEFAULT_MAX_SIZE);
    int* colonPositions          = (int*)malloc(sizeof(int) * T0_DEFAULT_MAX_SIZE);
    int* commaPositions          = (int*)malloc(sizeof(int) * T0_DEFAULT_MAX_SIZE);

    int* d_openBracePositions;
    int* d_closeBracePositions;
    int* d_openBracketPositions;
    int* d_closeBracketPositions;
    int* d_colonPositions;
    int* d_commaPositions;

    size_t json_size = json_content.size();

    int num_block = (json_size + NUM_PER_BLOCK * CHAR_PER_THREAD) / ((NUM_PER_BLOCK) * (CHAR_PER_THREAD));

    cudaMalloc((void**) &d_json_content, json_size);
    cudaMalloc((void**) &d_openBracePositions, sizeof(int) * T0_DEFAULT_MAX_SIZE);
    cudaMalloc((void**) &d_openBracketPositions, sizeof(int) * T0_DEFAULT_MAX_SIZE);
    cudaMalloc((void**) &d_closeBracePositions, sizeof(int) * T0_DEFAULT_MAX_SIZE);
    cudaMalloc((void**) &d_closeBracketPositions, sizeof(int) * T0_DEFAULT_MAX_SIZE);
    cudaMalloc((void**) &d_colonPositions, sizeof(int) * T0_DEFAULT_MAX_SIZE);
    cudaMalloc((void**) &d_commaPositions, sizeof(int) * T0_DEFAULT_MAX_SIZE);

    cudaMemcpy(d_json_content, json_content.data(), json_size, cudaMemcpyHostToDevice);

    build_structural_index<<<num_block, NUM_PER_BLOCK>>> (
        d_json_content,
        json_size,    
        d_openBracePositions,  
        d_closeBracePositions,
        d_openBracketPositions,
        d_closeBracketPositions,
        d_colonPositions,
        d_commaPositions
    );

    // Get the internal array pointer from device (you need to expose this pointer, e.g. via a kernel or by returning it)
    cudaMemcpy(openBracePositions, d_openBracePositions, sizeof(int) * T0_DEFAULT_MAX_SIZE, cudaMemcpyDeviceToHost);

    // Debug: check for successful memory copy
    for (int i = 0; i < T0_DEFAULT_MAX_SIZE; i++) {
        printf("%d \n", openBracePositions[i]);
    }
        




    return EXIT_SUCCESS;
}