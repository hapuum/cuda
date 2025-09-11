// Parses Reliquary Archiver JSON file and fills in information for HSR relics and characters
// Reliquary repository found at: https://github.com/IceDynamix/reliquary-archiver/tree/main
// The parser itself is simple and mainly focuses on converting JSON file to structs to work with in C++.

// Process in 3 steps:
// 1. Parellel Scan JSON file to generate structural index arrays
// 2. Parellel parse JSON file based on structural index arrays to generate HSR Relic data
// 3. Parellel generate HSR Relic data to HSR Sim data

// includes and defines
#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include "cuda_hsr_sim.h"

#define FILEPATH "/home/hapuum/cuda_learn/resource/relic_data.json"
#define CHAR_PER_THREAD 4096    // 1 full memory page
#define NUM_PER_BLOCK 256       // makes 1 block = 1MiB


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
        __device__ T& operator[](size_t idx) { return data[idx]; }
        __device__ size_t size() const { return length; }
};

__global__
void
build_structural_index
(
    // input
    char* file_data, 
    size_t file_size,
    // output 
    size_t* open_brace_positions,
    size_t* close_brace_positions,
    size_t* open_bracket_positions,
    size_t* close_bracket_positions,
    size_t* colon_positions,
    size_t* comma_positions
) 
{
    // save index from last occurrence in a matrix, each row = each thread, index 0~5 in this order
    // later used for joining the vectors.
    __shared__ int a[blockDim.x][6]; 

    int open_brace_last_occurrence;
    int close_brace_last_occurrence;
    int open_bracket_last_occurrence;
    int close_bracket_last_occurrence;
    int colon_last_occurrence;
    int comma_last_occurrence;

    DeviceVector<int>* open_brace_vector = new DeviceVector<int>();
    DeviceVector<int>* close_brace_vector = new DeviceVector<int>();
    DeviceVector<int>* open_bracket_vector = new DeviceVector<int>();
    DeviceVector<int>* close_bracket_vector = new DeviceVector<int>();
    DeviceVector<int>* colon_vector = new DeviceVector<int>();
    DeviceVector<int>* comma_vector = new DeviceVector<int>();

    // iterate through each character of assigned section of file_data
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = 0; i < CHAR_PER_THREAD; i++) {
        // update occurrence counter
        open_brace_last_occurrence++;
        close_brace_last_occurrence++;
        open_bracket_last_occurrence++;
        close_bracket_last_occurrence++;
        colon_last_occurrence++;
        comma_last_occurrence++;

        char c = file_data[tid * CHAR_PER_THREAD + i];
        switch (c) {
            case '{':
                open_brace_vector->push_back(open_brace_last_occurrence);
                open_brace_last_occurrence = 0;
            case '}':
                close_brace_vector->push_back(close_brace_last_occurrence);
                close_brace_last_occurrence = 0;
            case '[':
                open_bracket_vector->push_back(open_bracket_last_occurrence);
                open_bracket_last_occurrence = 0;
            case ']':
                close_bracket_vector->push_back(close_bracket_last_occurrence);
                close_bracket_last_occurrence = 0;
            case ':':
                colon_vector->push_back(open_brace_last_occurrence);
                colon_last_occurrence = 0;
            case ',':
                comma_vector->push_back(open_brace_last_occurrence);
                comma_last_occurrence = 0;
        }
    }

    open_brace_vector->push_back(-1 * open_brace_last_occurrence);
    close_brace_vector->push_back(-1 * close_brace_last_occurrence);
    open_bracket_vector->push_back(-1 * open_bracket_last_occurrence);
    close_bracket_vector->push_back(-1 * close_bracket_last_occurrence);
    colon_vector->push_back(-1 * open_brace_last_occurrence);
    comma_vector->push_back(-1 * open_brace_last_occurrence);

    __syncthreads();
    // join vectors by reduction
}




/*
__global__ void buildStructuralIndex() {
    
}




void scan_json(const char* json_content, size_t file_size, 
                            size_t* openBracePositions, 
                            size_t* closeBracePositions,
                            size_t* openBracketPositions,
                            size_t* closeBracketPositions,
                            size_t* colonPositions,
                            size_t* commaPositions) {
    
    
    
    
                                // determine block size:


    const int thread_scan_length = 4096; // standard length of page
    const int thread_per_block = 256; // each block processes 1 MiB




    int num_block = (file_size + thread_per_block * thread_scan_length) / (thread_per_block * thread_scan_length);
    // launch kernel with: <<<num_block, threads_per_block>>>

    

    // determine appropriate size for {} [] , :
    // set max for how much of each symbol can be parsed per thread, then launch __shared__
    // with total size num_thread * max_symbol_per_thread. there is a bit of memory inefficiency
    // and sparsity but allows us to be safely work on this. VRAM is pretty big for this task.
    // in a sense its a 0-terminated 2D list, of vec<vec<size_t>> that we are trying to have in here.
    // if a thread parses more than max_symbol_per_thread, record to some another flag and print out
    // so this limit can be increased.


    // MAYBE ACTUALLY JUST NEED TO IMPLEMENT CUSTOM VECTOR CLASS????
    // this kinda doesnt work
    // Kernel code -- to be modified later. 

    const int MAX_SYMBOL_PER_THREAD = 100;
    size_t size_per_thread = sizeof(int) * MAX_SYMBOL_PER_THREAD;
    const size_t thread_scan_length = length / blockDim.x;

    __shared__ size_t* openBraceIndex;
    __shared__ size_t* closeBraceIndex;
    __shared__ size_t* openBracketIndex;
    __shared__ size_t* closeBracketIndex;
    __shared__ size_t* colonIndex;
    __shared__ size_t* commaIndex;
    // these should be allocated as array of length (max_symbol_per_thread * thread)
    



    int count = 0;
    int index = 0;

    while (count < MAX_SYMBOL_PER_THREAD && index < thread_scan_length) {
        char c = json_content[index + threadIdx.x * thread_scan_length];
        printf(c);
        switch (c) {
            case '{':
                
            case '}':

            case '[':

            case ']':

            case ':':

            case ',':

            
        }
    }

    // NOW REDUCE THIS ARRAY BASED ON FIRST VALUE AND LAST VALUE





    



    // launch kernel, which collects intermediate data for the thread block

    // coalesce scanned data from each block into one list and save to output parameters 
    // need to join the symbols that start and end across thread/blocks. might need to launch separate kernel or
    // cpu can do this task? Maybe.
}


int main() {
    std::cout << "HSR Reliquary Archiver JSON Parser" << std::endl;
    // load json file
    std::ifstream file(FILEPATH);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << FILEPATH << std::endl;
        return EXIT_FAILURE;
    }
    std::string json_content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();
    std::cout << "File loaded successfully. Size: " << json_content.size() << " bytes" << std::endl;

    // scan json file to generate structural indices
    char* d_json_content;
    size_t* openBracePositions;
    size_t* closeBracePositions;
    size_t* openBracketPositions;
    size_t* closeBracketPositions;
    size_t* colonPositions;
    size_t* commaPositions;

    size_t json_size = json_content.size();
    cudaMalloc((void**) &d_json_content, json_size);
    cudaMemcpy(d_json_content, json_content.data(), json_size, cudaMemcpyHostToDevice);
    
    scan_json(d_json_content, json_size,    
                openBracePositions,  
                closeBracePositions,
                openBracketPositions,
                closeBracketPositions,
                colonPositions,
                commaPositions);
    return EXIT_SUCCESS;
}

*/