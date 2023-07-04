#include <stdio.h>
#include <stdint.h>
#include <defs.h>
#include <perfcounter.h>
#include <mram.h>
#include <mram_unaligned.h>
#include <barrier.h>
#include <string.h>

#include "bloom_filters_common.h"
#include "bloom_filters_dpu.h"

//---------------------------------------------------------------------------//
// PREPARATION
//---------------------------------------------------------------------------//

#define MAX_BLOOM_DPU_SIZE (1 << MAX_BLOOM_DPU_SIZE2)

#define ITEMS_CACHE_SIZE 256

BARRIER_INIT(end_lookup_barrier, NR_TASKLETS);

// Input from host
// WRAM
__host uint64_t _dpu_size2;
__host enum BloomMode _mode;
__host uint64_t _nb_hash;
// MRAM
__mram_noinit uint64_t items[MAX_NB_ITEMS_PER_DPU + 8];

// Own variables
// WRAM
uint64_t _dpu_size_reduced;
// MRAM
__mram_noinit uint8_t blooma[MAX_BLOOM_DPU_SIZE * NR_TASKLETS];
uint32_t nb_positive_lookups[NR_TASKLETS];

// Output to host
// WRAM - Performance
__host uint32_t nb_cycles;
__host uint32_t total_nb_positive_lookups;

//---------------------------------------------------------------------------//
// UTILITIES
//---------------------------------------------------------------------------//

static void insert_atomic(uint8_t *i, uint8_t args) { (*i) |= args; }

uint64_t simplehash16_64(uint64_t item, int idx) {
	uint64_t input = item >> idx;
	uint64_t res = random_values[input & 255];
	input = input >> 8;
	res  ^= random_values[input & 255];
	return res;
}

bool contains(uint64_t item, uint8_t __mram_ptr* the_blooma) {
	for (size_t k = 0; k < _nb_hash; k++) {
		uint64_t h0 = simplehash16_64(item, k) & _dpu_size_reduced;
		if ((the_blooma[h0 >> 3] & bit_mask[h0 & 7]) == 0) {  return false; }
	}
	return true;
}

//---------------------------------------------------------------------------//
// MAIN
//---------------------------------------------------------------------------//

int main() {

	uint8_t __mram_ptr* t_blooma = &blooma[MAX_BLOOM_DPU_SIZE * me()];

	perfcounter_config(COUNT_CYCLES, true);

	switch (_mode) {

		case INIT: {

			memset(t_blooma, 0, ((1 << _dpu_size2) >> 3) * sizeof(unsigned char));
			if (me() == 0) {
				dpu_printf("Filter size2 is = %lu\n", _dpu_size2);
				_dpu_size_reduced = (1 << _dpu_size2) - 1;
			}
			// dpu_printf("[%02d] My Bloom start adress is %p\n", me(), t_blooma);
			break;
		
		} case INSERT: {

			int t_items_cnt = 0;

			uint64_t items_cache[ITEMS_CACHE_SIZE];
			mram_read(items, items_cache, ITEMS_CACHE_SIZE * sizeof(uint64_t));
			uint64_t nb_items = items_cache[0];
			dpu_printf_0("We have %lu items\n", nb_items);
			int cache_idx = 1;
			for (int i = 0; i < nb_items; i++) {
				if (cache_idx == ITEMS_CACHE_SIZE) {
					mram_read(&items[i + 1], items_cache, ITEMS_CACHE_SIZE * sizeof(uint64_t));
					cache_idx = 0;
				}
				uint64_t item = items_cache[cache_idx];
				if ((item & 15) == me()) {
					for (size_t k = 0; k < _nb_hash; k++) {
						uint64_t h0 = simplehash16_64(item, k) & _dpu_size_reduced;
						t_blooma[h0 >> 3] |= bit_mask[h0 & 7];
						// mram_update_byte_atomic(&t_blooma[me()][h0 >> 3], insert_atomic, bit_mask[h0 & 7]); // Each tasklet has its own filter, no need to sync
					}
					t_items_cnt++;
				}
				cache_idx++;
			}
			dpu_printf_me("I have %d items\n", t_items_cnt);
			break;
		
		} case LOOKUP: {
			
			nb_positive_lookups[me()] = 0;
			uint64_t items_cache[ITEMS_CACHE_SIZE];
			mram_read(items, items_cache, ITEMS_CACHE_SIZE * sizeof(uint64_t));
			uint64_t nb_items = items_cache[0];
			int cache_idx = 1;
			for (int i = 0; i < nb_items; i++) {
				if (cache_idx == ITEMS_CACHE_SIZE) {
					mram_read(&items[i + 1], items_cache, ITEMS_CACHE_SIZE * sizeof(uint64_t));
					cache_idx = 0;
				}
				uint64_t item = items_cache[cache_idx];
				if ((item & 15) == me()) {
					bool result = contains(item, t_blooma);
					nb_positive_lookups[me()] += result;
					// dpu_printf("%d", result);
				}
				cache_idx++;
			}

			barrier_wait(&end_lookup_barrier);
			
			if (me() == 0) {
				// dpu_printf("\n");
				total_nb_positive_lookups = 0;
				for (size_t t = 0; t < NR_TASKLETS; t++) {
					total_nb_positive_lookups += nb_positive_lookups[t];
				}
			}
			break;
		
		}
	}

    nb_cycles = perfcounter_get();
    
    return 0;
}