#define MAX_NB_ITEMS_PER_DPU (1 << 16)
#define MAX_BLOOM_DPU_SIZE2 20

// #define LOG_DPU

#ifdef LOG_DPU
    #define dpu_printf(...) printf(__VA_ARGS__)
    #define dpu_printf_me(fmt, ...) printf("[%02d] " fmt, me(), ##__VA_ARGS__)
    #define dpu_printf_0(...) if (me() == 0) { dpu_printf_me(__VA_ARGS__); }
#else
    #define dpu_printf(...)
    #define dpu_printf_me(...)
    #define dpu_printf_0(...)
#endif

enum BloomMode {
    BLOOM_INIT,
    BLOOM_WEIGHT,
    BLOOM_INSERT,
    BLOOM_LOOKUP,
};

#define CEIL8(x) (((x) >> 3) << 3)