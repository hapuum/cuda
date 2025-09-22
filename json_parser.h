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
        LIST
    } json_type;

    class payload;

    class json_object {
        public:
        std::map<std::string, payload> data;
        
        json_object();

        void addObject(std::string key, int* val);
        void addObject(std::string key, double* val);
        void addObject(std::string key, std::string* val);
        void addObject(std::string key, bool* val);
        void addObject(std::string key, json_object* val);
        void addObject(std::string key, std::vector<payload>* val);

        ~json_object();

        payload operator[](std::string key);
    };

    class payload {
        public:
        void* value;
        json_type type;
        payload() : value(nullptr), type(INTEGER) { }
        payload(void* val, json_type t) : value(val), type(t) {};
    };

    class task {
        public:
        int start;
        int end;
        json_object* obj;

        task(int s, int e, json_object* o) : start(s), end(e), obj(o) {};
        ~task() {
            delete obj;
        }

        void setStart(int s);
        void setEnd(int e);
        void setObject(json_object* o);
    };
}