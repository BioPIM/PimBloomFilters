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

#define CACHE8_SIZE 2048
#define CACHE64_SIZE (CACHE8_SIZE >> 3)

// Barriers to sync tasklets
BARRIER_INIT(all_tasklets_barrier_1, NR_TASKLETS);
BARRIER_INIT(all_tasklets_barrier_2, NR_TASKLETS);

// Input from host
__host enum BloomFunction fctcall_id;
__mram_noinit struct BloomInfo init_data;
__mram_noinit uint64_t items[MAX_NB_ITEMS_PER_DPU + 1];
// __mram_noinit size_t dpu_uid;

// Own variables
// WRAM
__dma_aligned uint8_t gcache8[CACHE8_SIZE]; // Cache common to all tasklets
uint64_t* gcache64 = (uint64_t*) gcache8;
uint64_t _tasklet_results[NR_TASKLETS];
// MRAM
__mram_noinit uint8_t _bloom_data[MAX_BLOOM_DPU_SIZE * NR_TASKLETS];

// Output to host
__mram_noinit uint64_t result;

/* -------------------------------------------------------------------------- */
/*                                  Utilities                                 */
/* -------------------------------------------------------------------------- */

static void insert_atomic(uint8_t *i, uint8_t args) { (*i) |= args; }

static inline uint64_t simplehash16_64(uint64_t item, size_t idx) {
	uint64_t input = item >> idx;
	uint64_t res = random_values[input & 255];
	input = input >> 8;
	res  ^= random_values[input & 255];
	return res;
}

static inline void reduce_all_results() {
	barrier_wait(&all_tasklets_barrier_1);
	if (me() == 0) {
		uint64_t wram_result = 0;
		for (size_t t = 0; t < NR_TASKLETS; t++) {
			wram_result += _tasklet_results[t];
		}
		result = wram_result;
	}
}

/* -------------------------------------------------------------------------- */
/*                                    Main                                    */
/* -------------------------------------------------------------------------- */

int main() {

	struct BloomInfo wram_init_data = init_data;

	uint8_t __mram_ptr* _bloom_tasklet_data = &_bloom_data[MAX_BLOOM_DPU_SIZE * me()];
	size_t _bloom_nb_bytes = ((1UL << wram_init_data.dpu_size2) >> 3UL);

	__dma_aligned uint8_t cache8[CACHE8_SIZE]; // Local cache for each tasklet
	uint64_t* cache64 = (uint64_t*) cache8;

	// perfcounter_config(COUNT_CYCLES, true);

	// dpu_printf_0("Mode is %d\n", mode);

	switch(fctcall_id) {

		case BLOOM_INIT: {

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
			
			dpu_printf_0("Filter size2 = %lu\n", dpu_size2);
			break;

		}
		
		case BLOOM_WEIGHT: {
		
			dpu_printf("Compute weight\n");
			_tasklet_results[me()] = 0;
			if (wram_init_data.dpu_size2 < 3) {
				mram_read(_bloom_tasklet_data, cache8, 8 * sizeof(uint8_t));
				_tasklet_results[me()] += __builtin_popcount(cache8[0] & ((1 << (wram_init_data.dpu_size2 + 1)) - 1));
			} else {
				for (size_t i = 0; i < _bloom_nb_bytes; i += CACHE8_SIZE) {
					mram_read(&_bloom_tasklet_data[i], cache8, CACHE8_SIZE * sizeof(uint8_t));
					for (size_t k = 0; (k < CACHE8_SIZE) & ((i + k) < _bloom_nb_bytes); k++) {
						_tasklet_results[me()] += __builtin_popcount(cache8[k]);
					}
				}
			}
			reduce_all_results();
			break;
			
		}
		
		case BLOOM_INSERT: {

			uint64_t _dpu_size_reduced = (1 << wram_init_data.dpu_size2) - 1;
				
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
					for (size_t k = 0; k < wram_init_data.nb_hash; k++) {
						uint64_t h0 = simplehash16_64(item, k) & _dpu_size_reduced;
						_bloom_tasklet_data[h0 >> 3] |= bit_mask[h0 & 7];
						// mram_update_byte_atomic(&_bloom_tasklet_data[me()][h0 >> 3], insert_atomic, bit_mask[h0 & 7]); // Each tasklet has its own filter, no need to sync
					}
				}
				cache_idx++;
			}
			break;
			
		}
		
		case BLOOM_LOOKUP: {

			uint64_t _dpu_size_reduced = (1 << wram_init_data.dpu_size2) - 1;

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
					barrier_wait(&all_tasklets_barrier_1); // Wait, the result needs to be written
					if (me() == 0) {
						mram_write(gcache64, &items[current_start_idx], CACHE64_SIZE * sizeof(uint64_t));
					}
					barrier_wait(&all_tasklets_barrier_2); // Can go again
					current_start_idx += CACHE64_SIZE;
					mram_read(&items[current_start_idx], cache64, CACHE64_SIZE * sizeof(uint64_t));
					cache_idx = 0;
				}
				uint64_t item = cache64[cache_idx];
				if ((item & 15) == me()) {
					bool lookup_result = true;
					for (size_t k = 0; k < wram_init_data.nb_hash; k++) {
						uint64_t h0 = simplehash16_64(item, k) & _dpu_size_reduced;
						if ((_bloom_tasklet_data[h0 >> 3] & bit_mask[h0 & 7]) == 0) { lookup_result = false; break; }
					}
					gcache64[cache_idx] = lookup_result; // Write result in a global cache
				}
				cache_idx++;
			}
			barrier_wait(&all_tasklets_barrier_1);
			if (me() == 0) {
				// Write last part
				mram_write(gcache64, &items[current_start_idx], CACHE64_SIZE * sizeof(uint64_t));
			}
			break;
		}
		
	}

    // nb_cycles = perfcounter_get();
    
    return 0;
}