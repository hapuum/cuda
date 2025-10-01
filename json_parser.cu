#include "json_parser.h"
#include <fstream>
#include <stack>
#include <vector>

using namespace json;
using namespace std;
using namespace device_utils;

#define FINAL_VECTOR_SIZE 100000
#define DEFAULT_VECTOR_SIZE 64
#define CHAR_PER_THREAD 4096
#define NUM_PER_BLOCK 256


// implementation for json_parser.h definitions
namespace json {
    void task::setStart(int s) {
        start = s;
    }

    void task::setEnd(int e) {
        end = e;
    }

    void task::setObject(json_object* o) {
        obj = o;
    }

    void addObject(json_object& json, string_buffer& strbuf, json_buffer& jsonbuf, list_buffer& listbuf, const json_data data) {
        if (json.size >= 16) return;
        json_type t = data.type;
        json.child_json[json.size].type = t;
        json_value v;
        switch(t) {
            case(STRING):
                v.string_buffer_index = strbuf.size;
                json.child_json[json.size].val = v;
                strbuf.size++;
                break;
            case(JSON):
                v.json_buffer_index = jsonbuf.size;
                json.child_json[json.size].val = v;
                jsonbuf.size++;
                break;
            case(LIST):
                v.list_buffer_index = listbuf.size;
                json.child_json[json.size].val = v;
                listbuf.size++;
                break;
            default:
                break;
        }
        strbuf.size++; // for key
        json.size++; // added an element
    }
}

// Assumes clean data -- no escaped string, no tokens inside of a string. reasonable expectation for hsr sim use case
__global__ void get_sorted_structural_tokens(const char* json_content, StructuralToken* tokens, int* tokens_size, const size_t& json_size) {
    size_t vector_capacity = (threadIdx.x == 0) ? FINAL_VECTOR_SIZE : DEFAULT_VECTOR_SIZE;
    device_vector<StructuralToken>* token_vector = new device_vector<StructuralToken>(vector_capacity);

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = 0; i < CHAR_PER_THREAD; i++) {
        int current_loc = idx * CHAR_PER_THREAD + i;
        char c = json_content[idx * CHAR_PER_THREAD + i];
        StructuralToken tok;
        switch (c) {
            case '{':
                tok.location = current_loc;
                tok.t = OPEN_BRACE;
                token_vector->push_back(tok);
                break;
            case '}':
                tok.location = current_loc;
                tok.t = CLOSE_BRACE;
                token_vector->push_back(tok);
                break;
            case '[':
                tok.location = current_loc;
                tok.t = OPEN_BRACKET;
                token_vector->push_back(tok);
                break;
            case ']':
                tok.location = current_loc;
                tok.t = CLOSE_BRACKET;
                token_vector->push_back(tok);
                break;
            case ':':
                tok.location = current_loc;
                tok.t = COLON;
                token_vector->push_back(tok);
                break;
            case ',':
                tok.location = current_loc;
                tok.t = COMMA;
                token_vector->push_back(tok);
                break;
        }
    }
    __syncthreads();

    __shared__ device_vector<StructuralToken>* vector_array[NUM_PER_BLOCK];
    vector_array[threadIdx.x] = token_vector;

    for (int step = 1; step < blockDim.x; step *= 2) {
        if (threadIdx.x % (2 * step) == 0) {
            vector_array[threadIdx.x]->join(vector_array[threadIdx.x + step]);
        }
        __syncthreads();
    }


    if (threadIdx.x < 1) {
        *tokens_size = token_vector->size();
        printf("token size: %d \n", *tokens_size);
        for (int i = 0; i < *tokens_size; i++) {
            if (i > FINAL_VECTOR_SIZE) {
                printf("result needs to be bigger than %d \n", FINAL_VECTOR_SIZE);
                break;
            }
            StructuralToken st = vector_array[0]->get(i);
            tokens[i] = st;
        }
    }

    __syncthreads();
    delete token_vector;
    return;
}

// initializes objects and assigns correct indices of the buffer index that each json object needs to track.
// : indicates key-value pair, so it should mark a string at the location of colon and threads work left to parse
// { go into new json scope (start index of json)
// [ go into new list scope (start index of list)
// } go out of current json scope (end index of json)
// ] go out of current list scope (end index of list)
// , add new item to list (no transfer of index, use to parse index)
void initialize_buffer_connections() {
    // flags and stacks for nested object management
    // flag = current scope, stack = saved scope 
    


}


//     stack<task*> task_stack;
//     stack<bool> inList_stack;
//     stack<vector<json_data>*> current_list_stack;
//     stack<int> gpu_index_stack;
//     bool inString = false;
//     bool inList = false;
//     int start = json_content.find_first_of('{');
//     json_object* current_json =  new json_object();
//     int current_gpu_index = 0;
//     task* current_task = new task(start, 0, current_json);
//     string current_string;
//     vector<json_data>* current_list;
//     size_t json_size = json_content.size();
//     for (int i = start; i < json_size; i++) {
//         char c = json_content[i];
//         if (inString && c != '"') current_string = current_string + c;
//         switch (c) {
//             case '{' : {  // push in current task to save it, start working on new one
//                 if (inString) break;
//                 task_stack.push(current_task);
//                 json_object* new_json = new json_object();
//                 if (!inList) {
//                     current_json->addObject(current_string, new_json);
//                 }
//                 else {  // list stores payloads, instead of pair<string,payload>
//                     json_data p = json_data(new_json, JSON);
//                     current_list->push_back(p);
//                 }
//                 current_json = new_json;
//                 current_task = new task(i, 0, current_json);
//                 break;
//             }
//             case '}' : {  // finish and dispatch current task, go back to latest task
//                 if (inString) break;
//                 current_task->end = i;
//                 task_dispatchable.push_back(current_task);
//                 current_task = task_stack.top();
//                 current_json = current_task->obj;
//                 task_stack.pop();
//                 break;
//             }
//             case '"' : {
//                 inString = !inString;
//                 if (inString) current_string = "";  // entering new string
//                 break;
//             }
//             case '[' : {
//                 inList_stack.push(inList);
//                 inList = true;
//                 current_list_stack.push(current_list);
//                 current_list = new vector<json_data>();
//                 current_json->addObject(current_string, current_list);
//                 cout << current_string << endl;
//                 break;
//             }
//             case ']' : {
//                 inList = inList_stack.top();
//                 inList_stack.pop();
//                 current_list = current_list_stack.top();
//                 current_list_stack.pop();
//                 break;
//             }
//             default:
//                 break;
//         }
//     }  
// }




int main() {
    bool testing = false;
    string filepath = (!testing) ? "/home/hapuum/cuda_learn/resource/relic_data.json" : "/home/hapuum/cuda_learn/resource/test.json";

    ifstream file(filepath);
    if (!file.is_open()) 
    {
        std::cerr << "Error opening file: " << filepath << std::endl;
        return EXIT_FAILURE;
    }
    string json_content((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
    size_t json_size = json_content.size();
    file.close();
    cout << "File loaded successfully. Size: " << json_content.size() << " bytes" << endl;

    StructuralToken* sorted_tokens = (StructuralToken*) malloc(sizeof(StructuralToken) * FINAL_VECTOR_SIZE);
    StructuralToken* d_sorted_tokens;
    char* d_json_content;
    cudaMalloc((void**) &d_json_content, json_size);
    cudaMalloc((void**) &d_sorted_tokens, sizeof(StructuralToken) * FINAL_VECTOR_SIZE);
    cudaMemcpy(d_json_content, json_content.data(), json_size, cudaMemcpyHostToDevice);

    int* token_count = (int*) malloc(sizeof(int));
    *token_count = 0;
    int* d_token_count;
    cudaMalloc((void**)&d_token_count, sizeof(int));
    // Initialize the counter on the device to 0
    cudaMemset(d_token_count, 0, sizeof(int)); 

    //int num_block = (json_size + NUM_PER_BLOCK * CHAR_PER_THREAD) / ((NUM_PER_BLOCK) * (CHAR_PER_THREAD));
    int num_block = 1;
    get_sorted_structural_tokens<<<num_block, NUM_PER_BLOCK>>> (d_json_content, d_sorted_tokens, d_token_count, json_size);
    cudaDeviceSynchronize();

    cudaMemcpy(token_count, d_token_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(sorted_tokens, d_sorted_tokens, sizeof(StructuralToken) * FINAL_VECTOR_SIZE, cudaMemcpyDeviceToHost);

    // Now, *token_count on the host has the correct value.
    cout << "Kernel found " << *token_count << " tokens." << endl;
    
    // 3. Now you can safely access the results.
    cout << "Displaying first 8 tokens:" << endl;
    for (int i = 0; i < *token_count && i < 8; i++) {
        // Assuming your StructuralToken `t` is an enum or int you want to see
        cout << "Token type: " << sorted_tokens[i].t << " at location: " << sorted_tokens[i].location << endl;
    }

    cout << "Total size: " << *token_count << endl;

    // check for kernel/device errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }



    // now remap task_dispatchable to nice array structure where each thread can access
    // might need to implement device_compatible_vector and device_compatible_stack for the actual kernel code
    
}
