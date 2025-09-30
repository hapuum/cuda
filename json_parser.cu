#include "json_parser.h"
#include <fstream>
#include <stack>
#include <vector>

using namespace json;
using namespace std;
using namespace device_utils;

// implementation for json_parser.h definitions
namespace json {
    json_object::json_object() {
        size = 0;
    }

    json_object::~json_object() {
    }

    void task::setStart(int s) {
        start = s;
    }

    void task::setEnd(int e) {
        end = e;
    }

    void task::setObject(json_object* o) {
        obj = o;
    }

    // debug purpose
    // std::ostream& operator<< (std::ostream& stream, const json_object& json);
    // std::ostream& operator<< (std::ostream& stream, const json_data p) {
    //     switch(p.type) {
    //         case(INTEGER): {
    //             int* address = static_cast<int*>(p.value);
    //             stream << *address;
    //             break;
    //         }
    //         case(DOUBLE): {
    //             stream << *(static_cast<double*>(p.value));
    //             break;
    //         }
    //         case(STRING): {
    //             stream << *(static_cast<std::string*>(p.value));
    //             break;
    //         }
    //         case(BOOLEAN): {
    //             bool val = *(static_cast<bool*>(p.value));
    //             stream << (val ? "true" : "false"); 
    //             break;
    //         }
    //         case(JSON): {
    //             stream << *(static_cast<json_object*>(p.value));
    //             break;
    //         }
    //         case(LIST): {
    //             auto vec = static_cast<std::vector<json_data>*>(p.value);
    //             stream << "[";
    //             for (size_t i = 0; i < vec->size(); ++i) {
    //                 if (i > 0) stream << ", ";
    //                 stream << (*vec)[i];
    //             }
    //             stream << "]";
    //             break;
    //         }
    //     }
    //     return stream;
    // }
    // std::ostream& operator<< (std::ostream& stream, const json_object& json) {
    //     stream << "{";
    //     bool first = true;
    //     for (const auto& pair : json.data) {
    //         if (!first) {
    //             stream << ", ";
    //         }
    //         stream << pair.first << ": ";
    //         stream << pair.second;
    //         first = false;
    //     }
    //     stream << "}";
    //     return stream;
    // }

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

namespace device_utils {

}

__global__ void get_sorted_tokens(const std::string& json_content, device_vector<StructuralToken> tokens) {

}



// void create_json_tree(const std::string& json_content, std::vector<task*>& task_dispatchable) {
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
    file.close();
    cout << "File loaded successfully. Size: " << json_content.size() << " bytes" << endl;


    // now remap task_dispatchable to nice array structure where each thread can access
    // might need to implement device_compatible_vector and device_compatible_stack for the actual kernel code
    
}
 