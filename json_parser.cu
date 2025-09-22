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
    payload p;
    p.type = INTEGER;
    p.value = val;
    data[key] = p;
}

void json_object::addObject(std::string key, double* val) {
    payload p;
    p.type = DOUBLE;
    p.value = val;
    data[key] = p;
}

void json_object::addObject(std::string key, std::string* val) {
    payload p;
    p.type = STRING;
    p.value = val;
    data[key] = p;
}

void json_object::addObject(std::string key, bool* val) {
    payload p;
    p.type = BOOLEAN;
    p.value = val;
    data[key] = p;
}

void json_object::addObject(std::string key, json_object* val) {
    payload p;
    p.type = JSON;
    p.value = val;
    data[key] = p;
}

void json_object::addObject(std::string key, std::vector<payload>* val) {
    payload p;
    p.type = LIST;
    p.value = val;
    data[key] = p;
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
    // json_object json1, json2, json3, json4;
    // vector<payload> collection;
    
    // int i = 5;
    // bool b = true;
    // json1.addObject("first", &i);
    // json1.addObject("second", &b);
    // string s = "example string";
    // json2.addObject("third", &s);

    // payload p1, p2;
    // p1.value = &i,
    // p1.type = INTEGER;
    // p2.value = &json1;
    // p2.type = JSON;
    // collection.push_back(p1);
    // collection.push_back(p2);
    // json3.addObject("fourth", &collection);
    // json4.addObject("fifth", &json2);
    // cout << json1 << endl;
    // cout << json2 << endl;
    // cout << json3 << endl;
    // cout << json4 << endl;

    bool testing = false;
    string filepath = (testing) ? "/home/hapuum/cuda_learn/resource/relic_data.json" : "/home/hapuum/cuda_learn/resource/test.json";

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
    stack<task*> task_stack;
    vector<task*> task_dispatchable;

    bool inString = false;
    bool inList = false;
    int start = json_content.find_first_of('{');
    task* t = new task(start, 0, new json_object());

    size_t json_size = json_content.size();
    for (int i = start + 1; i < json_size; i++) {
        char c = json_content[i];
        switch (c) {
            case '{' : {  // push in current task to save it, start working on new one
                if (!inString) {
                    task_stack.push(t);
                    t = new task(i, 0, new json_object());  // THIS object needs to be added to parent object too? or at least connected somehow
                }
                break;
            }
            case '}' : {  // finish and dispatch current task, go back to latest task
                if (!inString) {
                    t->end = i;
                    task_dispatchable.push_back(t);                
                    t = task_stack.top();
                    task_stack.pop();
                }
                break;
            }
            case '"' : {
                inString = !inString;
                break;
            }
            case '[' : {
                inList = !inList;
                break;
            }
            case ']' : {
                inList = !inList;
                break;
            }
            case ',' : {
                if (inList && !inString) {

                }
                break;
            }
        }
    }    

    // GPU parallel processes all of the task_dispatchable and fills in the json_object in each task
    // extract them for host use


}
