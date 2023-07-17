#ifdef LOG_DPU
    #define dpu_printf(...) printf(__VA_ARGS__)
    #define dpu_printf_me(fmt, ...) printf("[%02d] " fmt, me(), ##__VA_ARGS__)
    #define dpu_printf_0(...) if (me() == 0) { dpu_printf_me(__VA_ARGS__); }
#else
    #define dpu_printf(...)
    #define dpu_printf_me(...)
    #define dpu_printf_0(...)
#endif