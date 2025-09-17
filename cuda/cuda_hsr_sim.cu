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
#include <map>
#include "cuda_hsr_sim.h"

#define FILEPATH "/home/hapuum/cuda_learn/resource/relic_data.json"
#define TEST_FILEPATH "/home/hapuum/cuda_learn/resource/test.json"
#define CHAR_PER_THREAD 4096        // 1 full memory page
#define NUM_PER_BLOCK 256           // makes 1 block = 1MiB
#define T0_DEFAULT_MAX_SIZE 32768  // default size of DeviceVector at thread 0 vs rest.
#define REST_DEFAULT_MAX_SIZE 64  // thread 0 is reserved more space for reduction stage

typedef struct Payload
{
    JsonObject json;
    std::string type;
};

class
JsonObject
{
    std::map<std::string, Payload> map;
    
    public:
        JsonObject() {}
        ~JsonObject() {}

    JsonObject operator[](std::string s) {
        return map[s].json;
    }

    void addEntry(int data) {};
    void addEntry(bool data) {};
    void addEntry(std::string data) {};
    void addEntry(JsonObject data[]) {};
    void addEntry(JsonObject data) {};

};

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
    int* comma_positions,
    size_t* size_open_brace,
    size_t* size_close_brace,
    size_t* size_open_bracket,
    size_t* size_close_bracket,
    size_t* size_colon,
    size_t* size_comma
) 
{

    size_t vector_capacity = (threadIdx.x == 0) ? T0_DEFAULT_MAX_SIZE : REST_DEFAULT_MAX_SIZE;

    // @TODO: MAYBE Refactor DeviceVector to __shared__ priorityqueue interface, 
    //          where each thread adds to the shared priorityqueue in some safe locked manner
    //          Such implementation can remove 1) reduction stage, 2) sorting stage for pair of symbols
    //          This might require coming up with a struct/class that can be added to this priorityqueue, 
    //          that can be either: pair {} or pair [] or : or ,
    
    DeviceVector<int>* open_brace_vector = new DeviceVector<int>(vector_capacity);
    DeviceVector<int>* close_brace_vector = new DeviceVector<int>(vector_capacity);
    DeviceVector<int>* open_bracket_vector = new DeviceVector<int>(vector_capacity);
    DeviceVector<int>* close_bracket_vector = new DeviceVector<int>(vector_capacity);
    DeviceVector<int>* colon_vector = new DeviceVector<int>(vector_capacity);
    DeviceVector<int>* comma_vector = new DeviceVector<int>(vector_capacity);

    // iterate through each character of assigned section of file_data
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = 0; i < CHAR_PER_THREAD; i++) 
    {
        char c = file_data[idx * CHAR_PER_THREAD + i];
        switch (c) 
        {
            case '{':
                open_brace_vector->push_back(idx * CHAR_PER_THREAD + i);
                break;
            case '}':
                close_brace_vector->push_back(idx * CHAR_PER_THREAD + i);
                break;
            case '[':
                open_bracket_vector->push_back(idx * CHAR_PER_THREAD + i);
                break;
            case ']':
                close_bracket_vector->push_back(idx * CHAR_PER_THREAD + i);
                break;
            case ':':
                colon_vector->push_back(idx * CHAR_PER_THREAD + i);
                break;
            case ',':
                comma_vector->push_back(idx * CHAR_PER_THREAD + i);
                break;
        }
    }
    
    __syncthreads();
    
    // join vectors by reduction
    // initialize 2D array of pointers to vectors storing each thread's vectors

    __shared__ DeviceVector<int>* vector_array[NUM_PER_BLOCK][6];
    vector_array[threadIdx.x][0] = open_brace_vector;
    vector_array[threadIdx.x][1] = close_brace_vector;
    vector_array[threadIdx.x][2] = open_bracket_vector;
    vector_array[threadIdx.x][3] = close_bracket_vector;
    vector_array[threadIdx.x][4] = colon_vector;
    vector_array[threadIdx.x][5] = comma_vector;

    for (int step = 1; step < blockDim.x; step *= 2) 
    {
        if (threadIdx.x % (2 * step) == 0) 
        {
            for (int i = 0; i < 6; i++) 
            {
                vector_array[threadIdx.x][i]->join(vector_array[threadIdx.x + step][i]);
            }
        }
        __syncthreads();
    }
    
    // assign return values
    // work 6 threads for copying each array to output 
    switch (threadIdx.x) 
    {
        case 0:
            for (int i = 0; i < vector_array[0][threadIdx.x]->size(); i++) 
            {
                open_brace_positions[i] = vector_array[0][threadIdx.x]->get(i);
            }
            *size_open_brace = vector_array[0][threadIdx.x]->size();
            break;
        case 1:
            for (int i = 0; i < vector_array[0][threadIdx.x]->size(); i++) 
            {
                close_brace_positions[i] = vector_array[0][threadIdx.x]->get(i);
            }
            *size_close_brace = vector_array[0][threadIdx.x]->size();
            break;
        case 2:
            for (int i = 0; i < vector_array[0][threadIdx.x]->size(); i++) 
            {
                open_bracket_positions[i] = vector_array[0][threadIdx.x]->get(i);
            }
            *size_open_bracket = vector_array[0][threadIdx.x]->size();
            break;
        case 3:
            for (int i = 0; i < vector_array[0][threadIdx.x]->size(); i++) 
            {
                close_bracket_positions[i] = vector_array[0][threadIdx.x]->get(i);
            }
            *size_close_bracket = vector_array[0][threadIdx.x]->size();
            break;
        case 4:
            for (int i = 0; i < vector_array[0][threadIdx.x]->size(); i++) 
            {
                colon_positions[i] = vector_array[0][threadIdx.x]->get(i);
            }
            *size_colon = vector_array[0][threadIdx.x]->size();
            break;
        case 5:
            for (int i = 0; i < vector_array[0][threadIdx.x]->size(); i++) 
            {
                comma_positions[i] = vector_array[0][threadIdx.x]->get(i);
            }
            *size_comma = vector_array[0][threadIdx.x]->size();
            break;
        default: 
            break;
    }
}

int main() {
    std::cout << "HSR Reliquary Archiver JSON Parser" << std::endl;
    // load json file
    bool testing = 0;
    std::string filepath = (testing) ? TEST_FILEPATH : FILEPATH;

    std::ifstream file(filepath);
    if (!file.is_open()) 
    {
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

    size_t size_openBrace_vector;
    size_t size_closeBrace_vector;
    size_t size_openBracket_vector;
    size_t size_closeBracket_vector;
    size_t size_colon_vector;
    size_t size_comma_vector;

    int* d_openBracePositions;
    int* d_closeBracePositions;
    int* d_openBracketPositions;
    int* d_closeBracketPositions;
    int* d_colonPositions;
    int* d_commaPositions;

    size_t* d_size_openBrace_vector;
    size_t* d_size_closeBrace_vector;
    size_t* d_size_openBracket_vector;
    size_t* d_size_closeBracket_vector;
    size_t* d_size_colon_vector;
    size_t* d_size_comma_vector;

    size_t json_size = json_content.size();

    int num_block = (json_size + NUM_PER_BLOCK * CHAR_PER_THREAD) / ((NUM_PER_BLOCK) * (CHAR_PER_THREAD));

    cudaMalloc((void**) &d_json_content, json_size);
    cudaMalloc((void**) &d_openBracePositions, sizeof(int) * T0_DEFAULT_MAX_SIZE);
    cudaMalloc((void**) &d_openBracketPositions, sizeof(int) * T0_DEFAULT_MAX_SIZE);
    cudaMalloc((void**) &d_closeBracePositions, sizeof(int) * T0_DEFAULT_MAX_SIZE);
    cudaMalloc((void**) &d_closeBracketPositions, sizeof(int) * T0_DEFAULT_MAX_SIZE);
    cudaMalloc((void**) &d_colonPositions, sizeof(int) * T0_DEFAULT_MAX_SIZE);
    cudaMalloc((void**) &d_commaPositions, sizeof(int) * T0_DEFAULT_MAX_SIZE);

    cudaMalloc((void**)&d_size_openBrace_vector, sizeof(size_t));
    cudaMalloc((void**)&d_size_closeBrace_vector, sizeof(size_t));
    cudaMalloc((void**)&d_size_openBracket_vector, sizeof(size_t));
    cudaMalloc((void**)&d_size_closeBracket_vector, sizeof(size_t));
    cudaMalloc((void**)&d_size_colon_vector, sizeof(size_t));
    cudaMalloc((void**)&d_size_comma_vector, sizeof(size_t));

    cudaMemcpy(d_json_content, json_content.data(), json_size, cudaMemcpyHostToDevice);

    build_structural_index<<<num_block, NUM_PER_BLOCK>>> (
        d_json_content,
        json_size,    
        d_openBracePositions,  
        d_closeBracePositions,
        d_openBracketPositions,
        d_closeBracketPositions,
        d_colonPositions,
        d_commaPositions,
        d_size_openBrace_vector,
        d_size_closeBrace_vector,
        d_size_openBracket_vector,
        d_size_closeBracket_vector,
        d_size_colon_vector,
        d_size_comma_vector
    );
    
    cudaMemcpy(openBracePositions, d_openBracePositions, sizeof(int) * T0_DEFAULT_MAX_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(closeBracePositions, d_closeBracePositions, sizeof(int) * T0_DEFAULT_MAX_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(openBracketPositions, d_openBracketPositions, sizeof(int) * T0_DEFAULT_MAX_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(closeBracketPositions, d_closeBracketPositions, sizeof(int) * T0_DEFAULT_MAX_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(colonPositions, d_colonPositions, sizeof(int) * T0_DEFAULT_MAX_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(commaPositions, d_commaPositions, sizeof(int) * T0_DEFAULT_MAX_SIZE, cudaMemcpyDeviceToHost);

    cudaMemcpy(&size_openBrace_vector, d_size_openBrace_vector, sizeof(size_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&size_closeBrace_vector, d_size_closeBrace_vector, sizeof(size_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&size_openBracket_vector, d_size_openBracket_vector, sizeof(size_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&size_closeBracket_vector, d_size_closeBracket_vector, sizeof(size_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&size_colon_vector, d_size_colon_vector, sizeof(size_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&size_comma_vector, d_size_comma_vector, sizeof(size_t), cudaMemcpyDeviceToHost);

    for (int i = 0; i < size_openBrace_vector; i++) {
        std::cout << openBracePositions[i] << std::endl;
    }


    std::cout << "size of open brace vector: " << size_openBrace_vector << std::endl;

    if (size_openBrace_vector != size_closeBrace_vector) {
        std::cout << "invalid file content: { and } amount does not match" << std::endl;
        return EXIT_FAILURE;
    }

    if (size_openBracket_vector != size_closeBracket_vector) {
        std::cout << "invalid file content: [ and ] amount does not match" << std::endl;
        return EXIT_FAILURE;
    }

    size_t pos_CHARACTERS   = json_content.find("characters");
    size_t pos_RELICS       = json_content.find("relics");

    if (pos_CHARACTERS == std::string::npos || pos_RELICS == std::string::npos) {
        std::cout << "invalid file content: cannot find \"characters\" or \"relics\"" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "\"characters\" is found at: " << pos_CHARACTERS << std::endl;
    std::cout << "\"relics\" is found at: " << pos_RELICS << std::endl;

    // DO SOME PROCESSING TO FIND WHERE THE RANGE FOR THESE END?

    //     struct Task {
    //     int start;
    //     int end;
    //      };

    // __device__ Task workQueue[MAX_TASKS];
    // __device__ int queueHead = 0;

    // __global__ void parse_json(char* json, int* openBraces, int* closeBraces, int numObjects) {
    //     int tid = threadIdx.x + blockIdx.x * blockDim.x;
    //     while (true) {
    //         int taskIdx = atomicAdd(&queueHead, 1);
    //         if (taskIdx >= numObjects) break;
    //         Task t = workQueue[taskIdx];
    //         // Parse json[t.start ... t.end]
               // SEARCH FOR "VARIABLE"
               // SWITCH("VARIABLE"):
               //   case("characters"), case("relics"), remaining things we can simply skip.
    //         // If nested object found, atomicAdd to queueHead and add new Task
    //     }
    // }



    return EXIT_SUCCESS;
}