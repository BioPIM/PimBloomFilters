#include <stdio.h>
#include <stdint.h>
#include <defs.h>
#include <perfcounter.h>
#include <mram.h>
#include <mram_unaligned.h>
#include <barrier.h>
#include <string.h>

#include "bloom_filters_dpu.h"

//---------------------------------------------------------------------------//
// PREPARATION
//---------------------------------------------------------------------------//

#define MAX_BLOOM_DPU_SIZE (1 << MAX_BLOOM_DPU_SIZE2)

#define ITEMS_CACHE_SIZE 128

#define INIT_BLOOM_CACHE_SIZE 2048

BARRIER_INIT(reduce_all_barrier, NR_TASKLETS);

// Input from host
// WRAM
__host uint64_t _dpu_size2;
__host enum BloomMode _mode;
__host uint64_t _nb_hash;
// MRAM
__mram_noinit uint64_t items[MAX_NB_ITEMS_PER_DPU + 8];
__mram_noinit uint32_t items_keys[MAX_NB_ITEMS_PER_DPU + 8];

// Own variables
// WRAM
uint64_t _dpu_size_reduced;
// MRAM
__mram_noinit uint8_t blooma[MAX_BLOOM_DPU_SIZE * NR_TASKLETS];
uint32_t t_results[NR_TASKLETS];

// Output to host
// WRAM - Performance
__host uint32_t nb_cycles;
__host uint32_t result;

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

void reduce_all_results() {
	barrier_wait(&reduce_all_barrier);
	if (me() == 0) {
		result = 0;
		for (size_t t = 0; t < NR_TASKLETS; t++) {
			result += t_results[t];
		}
	}
}

//---------------------------------------------------------------------------//
// MAIN
//---------------------------------------------------------------------------//

int main() {

	uint8_t __mram_ptr* t_blooma = &blooma[MAX_BLOOM_DPU_SIZE * me()];

	perfcounter_config(COUNT_CYCLES, true);

	dpu_printf_0("Mode is %d\n", _mode);

	switch (_mode) {

		case BLOOM_INIT: {

			// Basic version: memset in mram, a bit slow
			// memset(t_blooma, 0, ((1 << _dpu_size2) >> 3) * sizeof(unsigned char));

			// Better version: use a cache filled with zeros in wram
			uint8_t local_zeros[INIT_BLOOM_CACHE_SIZE];
			uint64_t total_size = ((1 << _dpu_size2) >> 3);
			memset(local_zeros, 0, INIT_BLOOM_CACHE_SIZE * sizeof(unsigned char));
			for (int i = 0; i < total_size; i += INIT_BLOOM_CACHE_SIZE) {
				if ((i + INIT_BLOOM_CACHE_SIZE) >= total_size) {
					mram_write(local_zeros, &t_blooma[i], CEIL8(total_size - i) * sizeof(unsigned char));
				} else {
					mram_write(local_zeros, &t_blooma[i], INIT_BLOOM_CACHE_SIZE * sizeof(unsigned char));
				}
			}
			
			if (me() == 0) {
				dpu_printf("Filter size2 = %lu\n", _dpu_size2);
				_dpu_size_reduced = (1 << _dpu_size2) - 1;
			}
			// dpu_printf("[%02d] My Bloom start adress is %p\n", me(), t_blooma);

			break;
		
		} case BLOOM_WEIGHT: {

			t_results[me()] = 0;
			uint8_t bloom_cache[INIT_BLOOM_CACHE_SIZE];
			if (_dpu_size2 < 3) {
				mram_read(t_blooma, bloom_cache, 8 * sizeof(unsigned char));
				t_results[me()] += __builtin_popcount(bloom_cache[0] & ((1 << (_dpu_size2 + 1)) - 1));
			} else {
				uint64_t total_size = ((1 << _dpu_size2) >> 3);
				for (int i = 0; i < total_size; i += INIT_BLOOM_CACHE_SIZE) {
					mram_read(&t_blooma[i], bloom_cache, INIT_BLOOM_CACHE_SIZE * sizeof(unsigned char));
					for (int k = 0; (k < INIT_BLOOM_CACHE_SIZE) & ((i + k) < total_size); k++) {
						t_results[me()] += __builtin_popcount(bloom_cache[k]);
					}
				}
			}
			reduce_all_results();
			break;
		
		} case BLOOM_INSERT: {

			int t_items_cnt = 0;

			__dma_aligned uint64_t items_cache[ITEMS_CACHE_SIZE];
			// __dma_aligned uint32_t items_keys_cache[ITEMS_CACHE_SIZE];
			mram_read(items, items_cache, ITEMS_CACHE_SIZE * sizeof(uint64_t));
			// mram_read(items_keys, items_keys_cache, ITEMS_CACHE_SIZE * sizeof(uint32_t));
			uint64_t nb_items = items_cache[0];
			// uint64_t nb_items = MAX_NB_ITEMS_PER_DPU;
			dpu_printf_0("We have %lu items\n", nb_items);
			int cache_idx = 1;
			for (int i = 0; i < nb_items; i++) {
				if (cache_idx == ITEMS_CACHE_SIZE) {
					mram_read(&items[i + 1], items_cache, ITEMS_CACHE_SIZE * sizeof(uint64_t));
					// mram_read(&items_keys[i + 1], items_keys_cache, ITEMS_CACHE_SIZE * sizeof(uint32_t));
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
			// dpu_printf_me("I have %d items\n", t_items_cnt);
			break;
		
		} case BLOOM_LOOKUP: {

			int t_items_cnt = 0;
			
			t_results[me()] = 0;
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
					bool result = contains(item, t_blooma);
					t_results[me()] += result;
					// dpu_printf("%d", result);
					t_items_cnt++;
				}
				cache_idx++;
			}
			dpu_printf_me("I have %d items\n", t_items_cnt);
			reduce_all_results();
			break;
		
		}
	}

    nb_cycles = perfcounter_get();
    
    return 0;
}