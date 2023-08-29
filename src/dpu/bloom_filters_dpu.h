#ifndef BAA0DC56_5396_444D_BF61_18C9AB62629A
#define BAA0DC56_5396_444D_BF61_18C9AB62629A

#include <stdint.h>
#include "pim_bloom_filter_common.h"
#include "dpu_utils.h"

const uint8_t bit_mask[8] = {
    0x01,  //00000001
    0x02,  //00000010
    0x04,  //00000100
    0x08,  //00001000
    0x10,  //00010000
    0x20,  //00100000
    0x40,  //01000000
    0x80   //10000000
};


#endif /* BAA0DC56_5396_444D_BF61_18C9AB62629A */
