#define CEIL8(x) ((((x) + 7) >> 3) << 3)

#define MAX_NB_ITEMS_PER_DPU (1 << 12)
#define MAX_BLOOM_DPU_SIZE2 20

enum BloomMode {
    BLOOM_INIT = 0,
    BLOOM_WEIGHT = 1,
    BLOOM_INSERT = 2,
    BLOOM_LOOKUP = 3,
};