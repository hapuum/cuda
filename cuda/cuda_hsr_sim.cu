// Parses Reliquary Archiver JSON file and fills in information for HSR relics and characters
// Reliquary repository found at: https://github.com/IceDynamix/reliquary-archiver/tree/main
// The parser itself is simple and mainly focuses on converting JSON file to structs to work with in C++.

// Process in 3 steps:
// 1. Parellel Scan JSON file to generate structural index arrays
// 2. Parellel parse JSON file based on structural index arrays to generate HSR Relic data
// 3. Parellel generate HSR Relic data to HSR Sim data

// includes and defines
#include <iostream>
#include <fstream>
#include <string>

#define FILEPATH "/home/hapuum/cuda_learn/resource/relic_data.json"

// THIS ENUM IS NOT YET FINISHED LIST OF ALL STATS, BUT IS SUFFICIENT FOR RELICS ONLY
// WILL EXPAND LATER WHEN CHARACTERS ARE IMPLEMENTED
typedef enum {
    HP_FLAT,
    HP_PCT,
    ATK_FLAT,
    ATK_PCT,
    DEF_FLAT,
    DEF_PCT,
    SPD,
    CRIT_RATE,
    CRIT_DMG,
    HEALING_BONUS,
    ENERGY_RECHARGE_RATE,
    EFFECT_HIT_RATE,
    EFFECT_RESISTANCE,
    ELEMENTAL_DAMAGE_PCT, // might need to expand to specific types in future
    STAT_TYPE_COUNT
} stat_type_t;

// typedef enum {} relic_set_t; // FUTURE USE CASE FOR RELIC SET EFFECTS

// typedefs
typedef struct {
    stat_type_t type;
    float value;
    int count;
    int step;
} substat_t;

typedef struct {
    int set_id;
    std::string set_name; // to be refactored to enum later
    std::string slot;
    int rarity;
    int level;
    stat_type_t main_stat;
    substat_t substats[4];
    std::string location;
    bool locked;
    bool discarded;
    int _uid;
} hsr_relic_t;

// FOR FUTURE USE IN SIMULATION. LEFT AS EMPTY FOR NOW.
// characters will most likely be stored in global memory as there are not many of them.
// as relic gets parsed, we can directly assign them to characters.
typedef struct {
    int basic;
    int skill;
    int ult;
    int talent;
    bool ability_1;
    bool ability_2;
    bool ability_3;
    bool stat_1;
    bool stat_2;
    bool stat_3;
    bool stat_4;
    bool stat_5;
    bool stat_6;
    bool stat_7;
    bool stat_8;
    bool stat_9;
    bool stat_10;
    int ability_version;
} skills_traces_t;

typedef struct {
    int id;
    std::string name;
    std::string path;
    int level;
    int ascension;
    int eidolon;
    skills_traces_t skills;
    hsr_relic_t relics[6];
} hsr_character_t;

void scan_json(const char* json_content, size_t length, 
                            size_t* openBracePositions, 
                            size_t* closeBracePositions,
                            size_t* openBracketPositions,
                            size_t* closeBracketPositions,
                            size_t* colonPositions,
                            size_t* commaPositions) {
    
    // determine block size:

    // determine appropriate size for {} [] , :
    // set max for how much of each symbol can be parsed per thread, then launch __shared__
    // with total size num_thread * max_symbol_per_thread. there is a bit of memory inefficiency
    // and sparsity but allows us to be safely work on this. VRAM is pretty big for this task.
    // in a sense its a 0-terminated 2D list, of vec<vec<size_t>> that we are trying to have in here.
    // if a thread parses more than max_symbol_per_thread, record to some another flag and print out
    // so this limit can be increased.

    // launch kernel, which collects intermediate data for the thread block

    // coalesce scanned data from each block into one list and save to output parameters 
    // need to join the symbols that start and end across thread/blocks. might need to launch separate kernel or
    // cpu can do this task? Maybe.
}


int main() {
    std::cout << "HSR Reliquary Archiver JSON Parser" << std::endl;
    // load json file
    std::ifstream file(FILEPATH);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << FILEPATH << std::endl;
        return EXIT_FAILURE;
    }
    std::string json_content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();
    std::cout << "File loaded successfully. Size: " << json_content.size() << " bytes" << std::endl;

    // scan json file to generate structural indices
    char* d_json_content;
    size_t* openBracePositions;
    size_t* closeBracePositions;
    size_t* openBracketPositions;
    size_t* closeBracketPositions;
    size_t* colonPositions;
    size_t* commaPositions;

    size_t json_size = json_content.size();
    cudaMalloc((void**) &d_json_content, json_size);
    cudaMemcpy(d_json_content, json_content.data(), json_size, cudaMemcpyHostToDevice);
    
    scan_json(d_json_content, json_size,    
                openBracePositions,  
                closeBracePositions,
                openBracketPositions,
                closeBracketPositions,
                colonPositions,
                commaPositions);
    return EXIT_SUCCESS;
}

