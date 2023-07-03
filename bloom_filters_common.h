#define MAX_NB_ITEMS_PER_DPU (1 << 10)
#define MAX_BLOOM_DPU_SIZE2 20

// #define LOG_DPU

#ifdef LOG_DPU
    #define dpu_printf(...) printf(__VA_ARGS__)
#else
    #define dpu_printf(...)
#endif

enum BloomMode {
    INIT,
    INSERT,
    LOOKUP,
};