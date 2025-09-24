#include "json_parser.h"
#include <fstream>
#include <stack>
#include <vector>

using namespace json;
using namespace std;

// implementation for json_parser.h definitions
namespace json {
    json_object::json_object() {
        data = std::map<std::string, payload>();
    }

    json_object::~json_object() {
    }

    void json_object::addObject(std::string key, int* val) {
        data[key] = payload(val, INTEGER);
    }

    void json_object::addObject(std::string key, double* val) {
        data[key] = payload(val, DOUBLE);
    }

    void json_object::addObject(std::string key, std::string* val) {
        data[key] = payload(val, STRING);
    }

    void json_object::addObject(std::string key, bool* val) {
        data[key] = payload(val, BOOLEAN);
    }

    void json_object::addObject(std::string key, json_object* val) {
        data[key] = payload(val, JSON);
    }

    void json_object::addObject(std::string key, std::vector<payload>* val) {
        data[key] = payload(val, LIST);
    }

    payload json_object::operator[](std::string key) {
        return data[key];
    }

    void task::setStart(int s) {
        start = s;
    }

    void task::setEnd(int e) {
        end = e;
    }

    void task:: setObject(json_object* o) {
        obj = o;
    }
    // debug purpose
    std::ostream& operator<< (std::ostream& stream, const json_object& json);
    std::ostream& operator<< (std::ostream& stream, const payload p) {
        switch(p.type) {
            case(INTEGER): {
                int* address = static_cast<int*>(p.value);
                stream << *address;
                break;
            }
            case(DOUBLE): {
                stream << *(static_cast<double*>(p.value));
                break;
            }
            case(STRING): {
                stream << *(static_cast<std::string*>(p.value));
                break;
            }
            case(BOOLEAN): {
                bool val = *(static_cast<bool*>(p.value));
                stream << (val ? "true" : "false"); 
                break;
            }
            case(JSON): {
                stream << *(static_cast<json_object*>(p.value));
                break;
            }
            case(LIST): {
                auto vec = static_cast<std::vector<payload>*>(p.value);
                stream << "[";
                for (size_t i = 0; i < vec->size(); ++i) {
                    if (i > 0) stream << ", ";
                    stream << (*vec)[i];
                }
                stream << "]";
                break;
            }
        }
        return stream;
    }
    std::ostream& operator<< (std::ostream& stream, const json_object& json) {
        stream << "{";
        bool first = true;
        for (const auto& pair : json.data) {
            if (!first) {
                stream << ", ";
            }
            stream << pair.first << ": ";
            stream << pair.second;
            first = false;
        }
        stream << "}";
        return stream;
    }
}

// CPU sequentially traverses json content string and finds pairs using stack, stores it in Task[]
// replace with structural index generation (parellel gpu) -> sort -> process the structural tokens from array
// instead of traversing entire json content string to make it more efficient 
void create_json_tree(const std::string& json_content, std::vector<task*>& task_dispatchable) {
    stack<task*> task_stack;
    stack<bool> inList_stack;
    stack<vector<payload>*> current_list_stack;

    bool inString = false;
    bool inList = false;
    int start = json_content.find_first_of('{');
    json_object* current_json =  new json_object();
    task* current_task = new task(start, 0, current_json);
    string current_string;
    vector<payload>* current_list;

    size_t json_size = json_content.size();
    for (int i = start; i < json_size; i++) {
        char c = json_content[i];
        if (inString && c != '"') current_string = current_string + c;
        switch (c) {
            case '{' : {  // push in current task to save it, start working on new one
                if (inString) break;
                task_stack.push(current_task);
                json_object* new_json = new json_object();
                if (!inList) {
                    current_json->addObject(current_string, new_json);

                }
                else {  // list stores payloads, instead of pair<string,payload>
                    payload p = payload(new_json, JSON);
                    current_list->push_back(p);
                }
                current_json = new_json;
                current_task = new task(i, 0, current_json);
                break;
            }
            case '}' : {  // finish and dispatch current task, go back to latest task
                if (inString) break;
                current_task->end = i;
                task_dispatchable.push_back(current_task);
                current_task = task_stack.top();
                current_json = current_task->obj;
                task_stack.pop();
                break;
            }
            case '"' : {
                inString = !inString;
                if (inString) current_string = "";  // entering new string
                break;
            }
            case '[' : {  // need better logic when adding nested array support
                inList_stack.push(inList);
                inList = true;
                current_list_stack.push(current_list);
                current_list = new vector<payload>();
                current_json->addObject(current_string, current_list);
                cout << current_string << endl;
                break;
            }
            case ']' : {
                inList = inList_stack.top();
                inList_stack.pop();
                current_list = current_list_stack.top();
                current_list_stack.pop();
                break;
            }
            default:
                break;
        }
    }  
}

// debug purpose and reference code for how to create / access different classes
void test_class_structure() {
    json_object json1, json2, json3, json4;
    vector<payload> collection;
    
    int i = 5;
    bool b = true;
    json1.addObject("first", &i);
    json1.addObject("second", &b);
    string s = "example string";
    json2.addObject("third", &s);

    payload p1(&i, INTEGER);
    payload p2(&json1, JSON);
    collection.push_back(p1);
    collection.push_back(p2);
    json3.addObject("fourth", &collection);
    json4.addObject("fifth", &json2);
    cout << json1 << endl;
    cout << json2 << endl;
    cout << json3 << endl;
    cout << json4 << endl;
}

__global__ void parse_json() {
    __shared__ extern char json_content[];

    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    
}

int main() {
    //test_class_structure();

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
    
    vector<task*> task_dispatchable;
    create_json_tree(json_content, task_dispatchable);

    task* d_task_array;
    cudaMalloc((void**) d_task_array, task.dispatchable.size() * sizeof(task*));
    // debug + prep for gpu
    for (int i = 0; i < task_dispatchable.size(); i++) {
        task* t = task_dispatchable[i];
        cout << "start :" << t->start << ", end:" << t->end << " json: " << *(t->obj) << " at memory: " << t->obj << endl;
    }


    // now remap task_dispatchable to nice array structure where each thread can access
    // might need to implement device_compatible_vector and device_compatible_stack for the actual kernel code
   
}
