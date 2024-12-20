#ifndef CD6575B7_00CE_4AC1_968C_A1BDEAE2E188
#define CD6575B7_00CE_4AC1_968C_A1BDEAE2E188

#include "pim_common.hpp"

#define CEIL8(x) ((((x) + 7) >> 3) << 3)

#define MAX_NB_ITEMS_PER_DPU (1 << 11) // 10 seems to be the best config (found empirically)
#define MAX_BLOOM_DPU_SIZE2 18
#define MAX_BLOOM_DPU_SIZE (1 << MAX_BLOOM_DPU_SIZE2)
#define CACHE8_BLOOM_SIZE 512
#define TOTAL_MAX_BLOOM_DPU_SIZE (MAX_BLOOM_DPU_SIZE + CACHE8_BLOOM_SIZE)

enum BloomFunction {
    BLOOM_INIT = 0,
    BLOOM_WEIGHT = 1,
    BLOOM_INSERT = 2,
    BLOOM_LOOKUP = 3,
};


#endif /* CD6575B7_00CE_4AC1_968C_A1BDEAE2E188 */
