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


// helper state enumeration to approach token processing as if it was an FSM
typedef enum {
    PROCESSING_JSON,
    PROCESSING_LIST,
    JSON_WAITING_INPUT
} token_parser_state;

inline std::ostream& operator<<(std::ostream& os, const token_parser_state& s) {
    switch (s) {
        case PROCESSING_JSON:     return os << "PROCESSING_JSON";
        case PROCESSING_LIST:     return os << "PROCESSING_LIST";
        case JSON_WAITING_INPUT:  return os << "JSON_WAITING_INPUT";
        default:                  return os << "UNKNOWN_STATE";
    }
}

// initializes objects and assigns correct indices of the buffer index that each json object needs to track.
// : indicates key-value pair, so it should mark a string at the location of colon and threads work left to parse
// { go into new json scope (start index of json)
// [ go into new list scope (start index of list)
// } go out of current json scope (end index of json)
// ] go out of current list scope (end index of list)
// , add new item to list (no transfer of index, use to parse index)
void initialize_buffer_connections
(
 StructuralToken* const tokens
,const int& tokens_size
,json::string_buffer& strbuf
,json::list_buffer& listbuf
,json::json_buffer& jsonbuf
) 
{
    if (tokens_size <= 0) return;

    // flags and stacks for nested object management
    // flag = current scope, stack = saved scope 

    stack<token_parser_state> state_stack;
    token_parser_state state;

    int local_index = 0;  // index within the current json / list scope
    stack<int> local_index_stack;

    int current_json_index = 0;  // index of current json object in global buffer
    stack<int> current_json_index_stack;

    int current_list_index = 0;  // index of current list object in global buffer
    stack<int> current_list_index_stack;


    if (tokens[0].t == OPEN_BRACE) state = PROCESSING_JSON;
    else if (tokens[0].t == OPEN_BRACKET) state = PROCESSING_LIST;
    else {
        printf("starting token invalid -- files need review before preceeding");
        return;
    } 
    // iterate through tokens and process in FSM style code
    for (int i = 1; i < tokens_size; i++) {
        StructuralToken tok = tokens[i];
        cout << "i = " << i << ", token location: " << tok.location << ", token type: " << tok.t;
        cout << ", state = " << state << endl;
        switch (state) {
            case PROCESSING_JSON: {
                if (tok.t != COLON) {
                    printf("error: processing json state expects a colon");
                }
                state = JSON_WAITING_INPUT;
                break;
            }
            case JSON_WAITING_INPUT: {
                switch (tok.t) {
                    case OPEN_BRACE:  // new json scope
                        state_stack.push(state);
                        state = PROCESSING_JSON;
                        local_index_stack.push(local_index);
                        local_index = 0;
                        current_json_index_stack.push(current_json_index);
                        current_json_index = 0; // THIS SHOULD BE IN CORRECT LOCATION OF THE CORRECT BUFFER, NOT 0
                        // CONNECT NEW JSON SCOPE AS CURRENT SCOPE'S CHILD
                        break;
                    case CLOSE_BRACE:  // this json object is done, close its scope and restore previous scope
                        if (state_stack.empty()) {
                                // closed the root JSON - parsing finished
                                cout << "closed root JSON, finishing parse loop" << endl;
                                // exit loop cleanly
                                i = tokens_size;
                                break;
                        }    
                        state = state_stack.top();
                        state_stack.pop(); 
                        local_index = local_index_stack.top();
                        local_index_stack.pop();
                        current_json_index = current_json_index_stack.top();
                        current_json_index_stack.pop();
                        break;
                    case OPEN_BRACKET: // new list scope
                        state_stack.push(state);
                        state = PROCESSING_LIST;
                        local_index_stack.push(local_index);
                        local_index = 0;
                        current_list_index_stack.push(current_list_index);
                        current_list_index = 0;  // THIS SHOULD BE IN CORRECT LOCATION OF THE CORRECT BUFFER, NOT 0
                        // CONNECT NEW LIST SCOPE AS CURRENT SCOPE'S CHILD
                        break;
                    case COMMA: // either primitive type or parsing this after restoring scope after finished list/json.
                                // increase local index and continue going
                        local_index++;
                        state = PROCESSING_JSON;
                        break;
                    default:
                        printf("error: invalid structural token type: %d\n", tok.t);
                        break;
                }
                break;
            }
            case PROCESSING_LIST: {
                switch (tok.t) {
                    case OPEN_BRACE:
                        state_stack.push(state);
                        state = PROCESSING_JSON;
                        local_index_stack.push(local_index);
                        local_index = 0;
                        current_json_index_stack.push(current_json_index);
                        current_json_index = 0; // THIS SHOULD BE IN CORRECT LOCATION OF THE CORRECT BUFFER, NOT 0
                        // link new children to current object before pushing
                        break;
                    case OPEN_BRACKET: 
                        state_stack.push(state);
                        state = PROCESSING_LIST;
                        local_index_stack.push(local_index);
                        local_index = 0;
                        current_list_index_stack.push(current_list_index);
                        current_list_index = 0; // THIS SHOULD BE IN CORRECT LOCATION OF THE CORRECT BUFFER, NOT 0
                        // link new children to current object before pushing
                        break;
                    case CLOSE_BRACKET:
                        if (state_stack.empty()) {
                            // closed the root list - finish parsing
                            cout << "closed root LIST, finishing parse loop" << endl;
                            i = tokens_size;
                            break;
                        }
                        state = state_stack.top();
                        state_stack.pop(); 
                        local_index = local_index_stack.top();
                        local_index_stack.pop();
                        current_list_index = current_list_index_stack.top();
                        current_list_index_stack.pop();
                        break;
                    case COMMA: // get next element
                        local_index++;
                        state = PROCESSING_LIST;
                        break;
                    default:
                        printf("error: invalid structural token type: %d at index %d \n", tok.t, tok.location);
                        break;
                }
                break;
            }
            default:
                printf("error: invalid state");
                break;
        }
    }
}

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

    string_buffer strbuf;
    list_buffer listbuf;
    json_buffer jsonbuf;

    strbuf.size = 0;
    listbuf.size = 0;
    jsonbuf.size = 0;


    initialize_buffer_connections(sorted_tokens, *token_count, strbuf, listbuf, jsonbuf);

    // now remap task_dispatchable to nice array structure where each thread can access
    // might need to implement device_compatible_vector and device_compatible_stack for the actual kernel code
    return EXIT_SUCCESS;
}
