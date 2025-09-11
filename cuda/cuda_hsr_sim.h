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