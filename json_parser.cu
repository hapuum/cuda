#include "json_parser.h"

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

void json_object::addObject(std::string key, std::vector<json_object>* val) {
    payload p;
    p.type = JSON_LIST;
    p.value = val;
    data[key] = p;
}

payload json_object::operator[](std::string key) {
    return data[key];
}

int main() {
    // TESTING CLASS STRUCTURE -- SUBJECT TO CHANGE AS WE MAKE IT DEVICE COMPATIBLE
    // STD::MAP and STD::VECTOR needs replacement
    //

    
    // using namespace std;
    // json_object json1, json2, json3, json4;
    // vector<json_object> collection;
    
    // int i = 5;
    // bool b = true;
    // json1.addObject("first", &i);
    // json1.addObject("second", &b);
    // string s = "example string";
    // json2.addObject("third", &s);
    // collection.push_back(json1);
    // collection.push_back(json2);
    // json3.addObject("fourth", &collection);
    // json4.addObject("fifth", &json2);
    // cout << json1 << endl;
    // cout << json2 << endl;
    // cout << json3 << endl;
    // cout << json4 << endl;

    // CPU sequentially traverses json content char[] and finds pairs using stack, stores it in Task[]
    // later to be replaced with a parellel algorithm (up sweep / down sweep based prefix sum pair generation)


    // GPU parallel processes all of the Task[] into json_object
}
