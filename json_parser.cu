#include "json_parser.h"
#include <fstream>
#include <stack>
#include <vector>

using namespace json;

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

int main() {
    using namespace std;
    // TEST CLASS STRUCTURE
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

    // CPU sequentially traverses json content char[] and finds pairs using stack, stores it in Task[]
    // later to be replaced with a parellel algorithm (up sweep / down sweep based prefix sum pair generation)
    // some efforts for parallelizing this part is in cuda/cuda_hsr_sim.cu, but needs more work for now.
    
    
    // string (variable name) also needs a stack of its own it seems.
    stack<task*> task_stack;
    vector<task*> task_dispatchable;

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
                inList = !inList;
                current_list = new vector<payload>();
                current_json->addObject(current_string, current_list);
                break;
            }
            case ']' : {
                inList = !inList;
                break;
            }
            case ',' : {
                if (inList && !inString) {
                    // push it to correct spot?? dont really need if we dont care about serialization.
                    // also the algorithm feels awkward on this... needs a separate stack to maintain if this is in a list or not
                }
                break;
            }
        }
    }    

    // debug + prep for gpu
    for (int i = 0; i < task_dispatchable.size(); i++) {
        task* t = task_dispatchable[i];
        cout << "start :" << t->start << ", end:" << t->end << " json :" << *(t->obj) << " at memory: " << t->obj << endl;
    }

    
    // GPU parallel processes all of the task_dispatchable and fills in the json_object in each task
    // extract them for host use


}
