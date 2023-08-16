#include <stdint.h>
#include <defs.h>
#include <perfcounter.h>
#include <mram.h>
#include <mram_unaligned.h>
#include <barrier.h>
#include <string.h>

#include "bloom_filters_dpu.h"

/* -------------------------------------------------------------------------- */
/*                                  Variables                                 */
/* -------------------------------------------------------------------------- */

#define MAX_BLOOM_DPU_SIZE (1 << MAX_BLOOM_DPU_SIZE2)

#define CACHE8_SIZE 2048
#define CACHE64_SIZE (CACHE8_SIZE >> 3)

BARRIER_INIT(reduce_all_barrier, NR_TASKLETS);
BARRIER_INIT(reduce_all_barrier2, NR_TASKLETS);

// Input from host
// WRAM
__host uint64_t dpu_size2;
__host enum BloomMode mode;
__host uint64_t nb_hash;
__host uint64_t dpu_uid;
// MRAM
__mram_noinit uint64_t items[MAX_NB_ITEMS_PER_DPU + 1];

// Own variables
// WRAM
uint64_t _dpu_size_reduced;
__dma_aligned uint8_t gcache8[CACHE8_SIZE];
uint64_t* gcache64 = (uint64_t*) gcache8;
// MRAM
__mram_noinit uint8_t _bloom_data[MAX_BLOOM_DPU_SIZE * NR_TASKLETS];
uint64_t _tasklet_results[NR_TASKLETS];

// Output to host
// WRAM - Performance
__host uint64_t nb_cycles;
__host uint64_t result;

/* -------------------------------------------------------------------------- */
/*                                  Utilities                                 */
/* -------------------------------------------------------------------------- */

static void insert_atomic(uint8_t *i, uint8_t args) { (*i) |= args; }

uint64_t simplehash16_64(uint64_t item, int idx) {
	uint64_t input = item >> idx;
	uint64_t res = random_values[input & 255];
	input = input >> 8;
	res  ^= random_values[input & 255];
	return res;
}

bool contains(uint64_t item, uint8_t __mram_ptr* data) {
	for (size_t k = 0; k < nb_hash; k++) {
		uint64_t h0 = simplehash16_64(item, k) & _dpu_size_reduced;
		if ((data[h0 >> 3] & bit_mask[h0 & 7]) == 0) {  return false; }
	}
	return true;
}

void reduce_all_results() {
	barrier_wait(&reduce_all_barrier);
	if (me() == 0) {
		result = 0;
		for (size_t t = 0; t < NR_TASKLETS; t++) {
			result += _tasklet_results[t];
		}
	}
}

/* -------------------------------------------------------------------------- */
/*                                    Main                                    */
/* -------------------------------------------------------------------------- */

int main() {

	uint8_t __mram_ptr* _bloom_tasklet_data = &_bloom_data[MAX_BLOOM_DPU_SIZE * me()];
	size_t _bloom_nb_bytes = ((1UL << dpu_size2) >> 3UL);

	__dma_aligned uint8_t cache8[CACHE8_SIZE];
	uint64_t* cache64 = (uint64_t*) cache8;

	perfcounter_config(COUNT_CYCLES, true);

	dpu_printf_0("Mode is %d\n", mode);

	if (mode == BLOOM_INIT) {

		// Basic version: memset in mram, a bit slow
		// memset(bloom_data, (uint8_t) 0, _bloom_nb_bytes * sizeof(uint8_t));

		// Better version: use a cache filled with zeros in wram
		memset(cache8, (uint8_t) 0, CACHE8_SIZE * sizeof(uint8_t));
		for (size_t i = 0; i < _bloom_nb_bytes; i += CACHE8_SIZE) {
			if ((i + CACHE8_SIZE) >= _bloom_nb_bytes) {
				mram_write(cache8, &_bloom_tasklet_data[i], CEIL8(_bloom_nb_bytes - i) * sizeof(uint8_t));
			} else {
				mram_write(cache8, &_bloom_tasklet_data[i], CACHE8_SIZE * sizeof(uint8_t));
			}
		}
		
		if (me() == 0) {
			dpu_printf("Filter size2 = %lu\n", dpu_size2);
			_dpu_size_reduced = (1 << dpu_size2) - 1;
		}

	} else if (mode == BLOOM_WEIGHT) {
		
		dpu_printf("Compute weight\n");
		_tasklet_results[me()] = 0;
		if (dpu_size2 < 3) {
			mram_read(_bloom_tasklet_data, cache8, 8 * sizeof(uint8_t));
			_tasklet_results[me()] += __builtin_popcount(cache8[0] & ((1 << (dpu_size2 + 1)) - 1));
		} else {
			for (size_t i = 0; i < _bloom_nb_bytes; i += CACHE8_SIZE) {
				mram_read(&_bloom_tasklet_data[i], cache8, CACHE8_SIZE * sizeof(uint8_t));
				for (size_t k = 0; (k < CACHE8_SIZE) & ((i + k) < _bloom_nb_bytes); k++) {
					_tasklet_results[me()] += __builtin_popcount(cache8[k]);
				}
			}
		}
		reduce_all_results();
		
	} else if (mode == BLOOM_INSERT) {
			
		mram_read(items, cache64, CACHE64_SIZE * sizeof(uint64_t));
		uint64_t _nb_items = cache64[0];
		
		// dpu_printf_0("We have %lu items\n", _nb_items);
		// if (_nb_items > MAX_NB_ITEMS_PER_DPU) {
		// 	halt(); // This should not happen, there is an error somewhere
		// }
		
		size_t cache_idx = 1;
		for (size_t i = 0; i < _nb_items; i++) {
			if (cache_idx == CACHE64_SIZE) {
				mram_read(&items[i + 1], cache64, CACHE64_SIZE * sizeof(uint64_t));
				cache_idx = 0;
			}
			uint64_t item = cache64[cache_idx];
			if ((item & 15) == me()) {
				for (size_t k = 0; k < nb_hash; k++) {
					uint64_t h0 = simplehash16_64(item, k) & _dpu_size_reduced;
					_bloom_tasklet_data[h0 >> 3] |= bit_mask[h0 & 7];
					// mram_update_byte_atomic(&_bloom_tasklet_data[me()][h0 >> 3], insert_atomic, bit_mask[h0 & 7]); // Each tasklet has its own filter, no need to sync
				}
			}
			cache_idx++;
		}
		
	} else if (mode == BLOOM_LOOKUP) {

		_tasklet_results[me()] = 0;

		mram_read(items, cache64, CACHE64_SIZE * sizeof(uint64_t));
		uint64_t _nb_items = cache64[0];
		
		// dpu_printf_0("We have %lu items\n", _nb_items);
		// if (_nb_items > MAX_NB_ITEMS_PER_DPU) {
		// 	halt(); // This should not happen, there is an error somewhere
		// }
		
		size_t cache_idx = 1;
		size_t current_start_idx = 0;
		for (size_t i = 0; i < _nb_items; i++) {
			if (cache_idx == CACHE64_SIZE) {
				barrier_wait(&reduce_all_barrier); // Wait, the result needs to be written
				if (me() == 0) {
					mram_write(gcache64, &items[current_start_idx], CACHE64_SIZE * sizeof(uint64_t));
				}
				barrier_wait(&reduce_all_barrier2); // Can go again
				current_start_idx += CACHE64_SIZE;
				mram_read(&items[current_start_idx], cache64, CACHE64_SIZE * sizeof(uint64_t));
				cache_idx = 0;
			}
			uint64_t item = cache64[cache_idx];
			if ((item & 15) == me()) {
				gcache64[cache_idx] = contains(item, _bloom_tasklet_data); // Write result in a global cache
			}
			cache_idx++;
		}
		barrier_wait(&reduce_all_barrier);
		if (me() == 0) {
			// Write last part
			mram_write(gcache64, &items[current_start_idx], CACHE64_SIZE * sizeof(uint64_t));
		}
		
	}

    nb_cycles = perfcounter_get();
    
    return 0;
}