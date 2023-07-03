#define MAX_NB_ITEMS_PER_DPU (1 << 10)
#define MAX_BLOOM_DPU_SIZE2 11

enum BloomMode {
    INIT,
    INSERT,
    LOOKUP,
};