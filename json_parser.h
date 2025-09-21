#pragma once
#include <map>
#include <string>
#include <vector>
#include <iostream>

namespace json {
    typedef enum {
        INTEGER,
        DOUBLE,
        STRING,
        BOOLEAN,
        JSON,
        JSON_LIST
    } json_type;

    typedef struct {
        void* value;
        json_type type;
    } payload;

    class json_object {
        public:
        std::map<std::string, payload> data;
        
        json_object();

        void addObject(std::string key, int* val);
        void addObject(std::string key, double* val);
        void addObject(std::string key, std::string* val);
        void addObject(std::string key, bool* val);
        void addObject(std::string key, json_object* val);
        void addObject(std::string key, std::vector<json_object>* val);

        ~json_object();

        payload operator[](std::string key);

        // debug purposes
        friend std::ostream& operator<< (std::ostream& stream, const json_object& json) {
            stream << "{";
            bool first = true;
            for (const auto& pair : json.data) {
                if (!first) {
                    stream << ", ";
                }
                stream << pair.first << ": ";
                switch(pair.second.type) {
                    case(INTEGER): {
                        int* address = static_cast<int*>(pair.second.value);
                        stream << *address;
                        break;
                    }
                    case(DOUBLE): {
                        stream << *(static_cast<double*>(pair.second.value));
                        break;
                    }
                    case(STRING): {
                        stream << *(static_cast<std::string*>(pair.second.value));
                        break;
                    }
                    case(BOOLEAN): {
                        bool val = *(static_cast<bool*>(pair.second.value));
                        stream << (val ? "true" : "false"); 
                        break;
                    }
                    case(JSON): {
                        stream << *(static_cast<json_object*>(pair.second.value));
                        break;
                    }
                    case(JSON_LIST): {
                        auto vec = static_cast<std::vector<json_object>*>(pair.second.value);
                        stream << "[";
                        for (size_t i = 0; i < vec->size(); ++i) {
                            if (i > 0) stream << ", ";
                            stream << (*vec)[i];
                        }
                        stream << "]";
                        break;
                    }
                }
                first = false;
            }
            stream << "}";
            return stream;
        }
    };

    class task {
        public:
        int start;
        int end;
        json_object obj;

        task(int s, int e, const json_object& o) : start(s), end(e), obj(o) {};
    };
}