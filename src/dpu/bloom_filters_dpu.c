#include <stdint.h>
#include <defs.h>
#include <perfcounter.h>
#include <mram.h>
#include <mram_unaligned.h>
#include <barrier.h>
#include <mutex_pool.h>
#include <string.h>

#include "bloom_filters_dpu.h"
#include "murmur3.h"


/* -------------------------------------------------------------------------- */
/*                                  Variables                                 */
/* -------------------------------------------------------------------------- */

#define CACHE8_SIZE 2048
#define CACHE64_SIZE (CACHE8_SIZE >> 3)
#define BLOCK_MODULO 4095 // Must be (CACHE8_BLOOM_SIZE * 8) - 1

// Barriers to sync tasklets
BARRIER_INIT(all_tasklets_barrier_1, NR_TASKLETS);
BARRIER_INIT(all_tasklets_barrier_2, NR_TASKLETS);

MUTEX_POOL_INIT(write_mutex, NR_TASKLETS);

// Input from host
__mram_noinit uint64_t args[MAX_NB_ITEMS_PER_DPU + 3];
// __mram_noinit size_t dpu_uid;

// Own variables
// WRAM
__dma_aligned uint8_t gcache8[CACHE8_SIZE]; // Cache common to all tasklets
uint64_t* gcache64 = (uint64_t*) gcache8;
uint64_t _tasklet_results[NR_TASKLETS];
// MRAM
__mram_noinit uint8_t _bloom_data[TOTAL_MAX_BLOOM_DPU_SIZE * NR_TASKLETS];
__mram_noinit uint64_t dpu_size2;
__mram_noinit uint64_t nb_hash;

// Output to host
__mram_noinit uint64_t result;
__mram_noinit uint64_t perf_counter;
__mram_noinit uint64_t perf_ref_id;


/* -------------------------------------------------------------------------- */
/*                                  Utilities                                 */
/* -------------------------------------------------------------------------- */

static inline uint64_t simplehash16_64(uint64_t item, size_t idx) {
	uint64_t i = item >> idx;
    uint16_t hash = (uint16_t)(i & 0xFFFF);
    hash ^= (uint16_t)((i >> 16) & 0xFFFF);
    hash ^= (uint16_t)((i >> 32) & 0xFFFF);
    hash ^= (uint16_t)((i >> 48) & 0xFFFF);
    return (uint64_t) hash;
}

static inline uint64_t hash64(uint64_t key) {
	uint64_t hash = 0;
	MurmurHash3_x86_32(&key, sizeof(uint64_t), 0, &hash);
	return hash;
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

	#ifdef DO_DPU_PERFCOUNTER
	#if(DO_DPU_PERFCOUNTER == 0)
		perfcounter_config(COUNT_CYCLES, true);
	#else
		perfcounter_config(COUNT_INSTRUCTIONS, true);
	#endif
	perf_ref_id = args[0];
	#endif


	/* ----------------------- Compute tasklet useful data ---------------------- */

	__mram_ptr uint8_t* _bloom_tasklets_data[NR_TASKLETS];
	for (size_t n = 0; n < NR_TASKLETS; n++) {
		_bloom_tasklets_data[n] = &_bloom_data[TOTAL_MAX_BLOOM_DPU_SIZE * n];
	}
	__mram_ptr uint8_t* _bloom_tasklet_data = _bloom_tasklets_data[me()];


	__dma_aligned uint8_t cache8[CACHE8_SIZE]; // Local cache for each tasklet
	uint64_t* cache64 = (uint64_t*) cache8;

	__dma_aligned uint8_t cache8_bloom[CACHE8_BLOOM_SIZE]; // Local cache for each tasklet

	mram_read(args, cache64, CACHE64_SIZE * sizeof(uint64_t));


	/* ------------------------------ Call function ----------------------------- */

	switch(cache64[0]) {

		case BLOOM_INIT: {

			size_t _bloom_nb_bytes = (1 << (cache64[1] - 3));
			// Store data for future calls
			if (me() == 0) { // Must be done before filling with 0s because cache64 will be overwritten
				dpu_size2 = cache64[1];
				nb_hash = cache64[2];
				dpu_printf("Filter size2 = %lu\n", dpu_size2);
			}

			// Basic version: memset in mram, a bit slow
			// memset(_bloom_tasklet_data, 0, _bloom_nb_bytes * sizeof(uint8_t));

			// Better version: use a cache filled with zeros in wram
			for (size_t i = 0; i < CACHE8_SIZE; i++) { cache8[i] = 0; }
			for (size_t i = 0; i < _bloom_nb_bytes; i += CACHE8_SIZE) {
				if ((i + CACHE8_SIZE) >= _bloom_nb_bytes) {
					mram_write(cache8, &_bloom_tasklet_data[i], CEIL8(_bloom_nb_bytes - i) * sizeof(uint8_t));
				} else {
					mram_write(cache8, &_bloom_tasklet_data[i], CACHE8_SIZE * sizeof(uint8_t));
				}
			}

			break;

		}
		
		case BLOOM_WEIGHT: {

			uint64_t wram_dpu_size2 = dpu_size2;
			size_t _bloom_nb_bytes = (1 << (dpu_size2 - 3));
		
			_tasklet_results[me()] = 0;
			if (wram_dpu_size2 < 3) {
				mram_read(_bloom_tasklet_data, cache8, 8 * sizeof(uint8_t));
				_tasklet_results[me()] += __builtin_popcount(cache8[0] & ((1 << (wram_dpu_size2 + 1)) - 1));
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

			uint64_t _dpu_size_reduced = (1 << dpu_size2) - 1;
			uint64_t wram_nb_hash = nb_hash;
			uint64_t _nb_items = cache64[1];

			size_t D = _nb_items >> 4;
			size_t K = _nb_items & 15;

			size_t start_index = me() * D + (me() < K ? me() : K);
			size_t stop_index = start_index + D + (me() < K ? 1 : 0);
			
			size_t cache_idx = CACHE64_SIZE;
			for (size_t i = start_index; i < stop_index; i++) {
				if (cache_idx >= CACHE64_SIZE) {
					mram_read(&args[i + 2], cache64, CACHE64_SIZE * sizeof(uint64_t));
					cache_idx = 0;
				}
				uint64_t item = cache64[cache_idx];
				uint64_t h0 = hash64(item) & _dpu_size_reduced;
				size_t filter_id = (simplehash16_64(item, 0) & 15);
				size_t h0_idx = (h0 >> 3);
				mutex_pool_lock(&write_mutex, filter_id);
				mram_read(&_bloom_tasklets_data[filter_id][h0_idx], cache8_bloom, CACHE8_BLOOM_SIZE * sizeof(uint8_t));
				cache8_bloom[0] |= bit_mask[h0 & 7];
				for (size_t k = 1; k < wram_nb_hash; k++) {
					uint64_t h1 = simplehash16_64(item, k) & BLOCK_MODULO;
					cache8_bloom[h1 >> 3] |= bit_mask[h1 & 7];
				}
				mram_write(cache8_bloom, &_bloom_tasklets_data[filter_id][h0_idx], CACHE8_BLOOM_SIZE * sizeof(uint8_t));
				mutex_pool_unlock(&write_mutex, filter_id);
				cache_idx++;
			}
			break;
			
		}
		
		case BLOOM_LOOKUP: {

			uint64_t _dpu_size_reduced = (1 << dpu_size2) - 1;
			uint64_t wram_nb_hash = nb_hash;
			_tasklet_results[me()] = 0;
			uint64_t _nb_items = cache64[1];
			
			// dpu_printf_0("We have %lu items\n", _nb_items);
			// if (_nb_items > MAX_NB_ITEMS_PER_DPU) {
			// 	halt(); // This should not happen, there is an error somewhere
			// }
			
			if (me() == 0) {
				gcache64[0] = _nb_items; // Write number of items in the result because the host will need it
			}
			size_t cache_idx = 2;
			size_t current_start_idx = 0;
			for (size_t i = 0; i < _nb_items; i++) {
				if (cache_idx == CACHE64_SIZE) {
					barrier_wait(&all_tasklets_barrier_1); // Wait, the result needs to be written
					if (me() == 0) {
						mram_write(gcache64, &args[current_start_idx], CACHE64_SIZE * sizeof(uint64_t));
					}
					barrier_wait(&all_tasklets_barrier_2); // Can go again
					current_start_idx += CACHE64_SIZE;
					mram_read(&args[current_start_idx], cache64, CACHE64_SIZE * sizeof(uint64_t));
					cache_idx = 0;
				}
				uint64_t item = cache64[cache_idx];
				if ((simplehash16_64(item, 0) & 15) == me()) {
					uint64_t h0 = hash64(item) & _dpu_size_reduced;
					size_t h0_idx = (h0 >> 3);
					mram_read(&_bloom_tasklet_data[h0_idx], cache8_bloom, CACHE8_BLOOM_SIZE * sizeof(uint8_t));
					bool lookup_result = !((cache8_bloom[0] & bit_mask[h0 & 7]) == 0);
					if (lookup_result) {
						for (size_t k = 1; k < wram_nb_hash; k++) {
							uint64_t h1 = simplehash16_64(item, k) & BLOCK_MODULO;
							if ((cache8_bloom[h1 >> 3] & bit_mask[h1 & 7]) == 0) { lookup_result = false; break; }
						}
					}
					gcache64[cache_idx] = lookup_result; // Write result in a global cache
				}

				cache_idx++;
			}
			barrier_wait(&all_tasklets_barrier_1);
			if (me() == 0) {
				// Write last part
				mram_write(gcache64, &args[current_start_idx], CACHE64_SIZE * sizeof(uint64_t));
			}
			break;
		}
		
	}

	#ifdef DO_DPU_PERFCOUNTER
    perf_counter = perfcounter_get();
	#endif
    
    return 0;
}