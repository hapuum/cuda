#pragma once
#include <map>
#include <string>
#include <vector>
#include <iostream>
#define MAX_CHILDREN_PER_JSON 16
#define MAX_STRINGS 16384
#define MAX_STRING_LENGTH 32
#define MAX_LISTS   16384
#define MAX_LIST_LENGTH 65536

namespace json {
    typedef enum {
        INTEGER,
        DOUBLE,
        STRING,
        BOOLEAN,
        JSON,
        LIST
    } json_type;

    typedef union {
        int i;
        double d;
        int string_buffer_index;
        bool b;
        int list_buffer_index;
        int json_buffer_index; 
    } json_value;

    class json_data;
    class string_buffer;
    class json_buffer;
    class list_buffer;

    class json_data {
        public:
        json_value val;
        json_type type;
    };

    class json_object {
        public:
        int key_string_indices[16];
        json_data child_json[16];
        int size;
    };



    // device compatible string buffer. index to this serves like a pointer.
    class string_buffer {
        public:
        char data[MAX_STRINGS][MAX_STRING_LENGTH];
        int start[MAX_STRINGS];  // location of associated colon or list delimiters
        int size;
        // string lengths are kind of hard to determine right off of tokens and does not vary much so 2D array makes sense
    };
    class json_buffer {
        public:
        json_object data[MAX_CHILDREN_PER_JSON];
        int start[MAX_CHILDREN_PER_JSON];
        int end[MAX_CHILDREN_PER_JSON];
        int size;
    };
    class list_buffer {
        public:
        json_data data[MAX_LIST_LENGTH];
        int list_starting_index[MAX_LISTS];
        int start[MAX_LISTS];
        int end[MAX_LISTS];
        int size;
        // Tokens are perfectly fine to generate starting index of each list,
        // where other objects navigate into list_starting_index then use that to get where to start data
    };

    // adds an object to json and modifies the global buffers as needed.
    // string buffer size will increase in all operations, and json's key will be set to new strbuf size
    // the string buffer may additionally increase when datatype is string
    // similar logic for jsonbuf / listbuf. increase its size and point to correct location as needed.
    void addObject(json_object& json, string_buffer& strbuf, json_buffer& jsonbuf, list_buffer& listbuf, const json_data data);

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
        void setGPUIndex(int g);
    };

    typedef enum {
        OPEN_BRACE,
        CLOSE_BRACE,
        OPEN_BRACKET,
        CLOSE_BRACKET,
        COLON,
        COMMA
    } tokenType;

    class StructuralToken {
        public:
        int location;
        tokenType t;
    };

    // ostream operator overloads so enums/structs print readable names during debug.
    inline std::ostream& operator<<(std::ostream& os, const tokenType& tt) {
        switch (tt) {
            case OPEN_BRACE:   return os << "OPEN_BRACE";
            case CLOSE_BRACE:  return os << "CLOSE_BRACE";
            case OPEN_BRACKET: return os << "OPEN_BRACKET";
            case CLOSE_BRACKET: return os << "CLOSE_BRACKET";
            case COLON:        return os << "COLON";
            case COMMA:        return os << "COMMA";
            default:           return os << "UNKNOWN_TOKEN";
        }
    }

    inline std::ostream& operator<<(std::ostream& os, const StructuralToken& st) {
        return os << "StructuralToken(type=" << st.t << ", location=" << st.location << ")";
    }
}

namespace device_utils {
    template<typename T> class device_vector {
        T* data;
        size_t capacity, length;
        public:
        __device__ device_vector() : data(nullptr), capacity(16), length(0) {
            data = new T[capacity];
        }

        __device__ device_vector(size_t c) : data(nullptr), capacity(c), length(0) {
            data = new T[capacity];
        }

        __device__ ~device_vector() { delete[] data; }

        __device__ void push_back(const T& value) {
            if (length >= capacity) {
                capacity *= 2;
                T* new_data = new T[capacity];
                for (size_t i = 0; i < length; ++i) new_data[i] = data[i];
                delete[] data;
                data = new_data;
            }
            data[length++] = value;
        }

        __device__ void set(size_t idx, const T& value) { data[idx] = value; }

        __device__ void join(const device_vector<T>* next) {
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
        __host__ __device__ T& const get(size_t idx) const { return data[idx]; }
    };
}
