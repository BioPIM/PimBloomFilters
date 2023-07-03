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
uint8_t __mram_ptr* blooma_starts[NR_TASKLETS];
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

bool contains(uint64_t item) {
	for (size_t k = 0; k < _nb_hash; k++) {
		uint64_t h0 = simplehash16_64(item, k) & _dpu_size_reduced;
		if ((blooma_starts[me()][h0 >> 3] & bit_mask[h0 & 7]) == 0) {  return false; }
	}
	return true;
}

//---------------------------------------------------------------------------//
// MAIN
//---------------------------------------------------------------------------//

int main() {

	if (me() == 0) {
		printf("DPU running\n");
	}

	perfcounter_config(COUNT_CYCLES, true);

	switch (_mode) {

		case INIT: {

			blooma_starts[me()] = &blooma[MAX_BLOOM_DPU_SIZE * me()];
			memset(blooma_starts[me()], 0, ((1 << _dpu_size2) >> 3) * sizeof(unsigned char));
			if (me() == 0) {
				printf("Filter size2 is = %lu\n", _dpu_size2);
				_dpu_size_reduced = (1 << _dpu_size2) - 1;
			}
			// printf("[%02d] My Bloom start adress is %p\n", me(), blooma_starts[me()]);
			break;
		
		} case INSERT: {

			uint64_t items_cache[ITEMS_CACHE_SIZE];
			mram_read(items, items_cache, ITEMS_CACHE_SIZE * sizeof(uint64_t));
			uint64_t nb_items = items[0];
			int item_cnt = 0;
			int cache_idx = 0;
			while (item_cnt < nb_items) {}
			for (int i = 0; i < nb_items; i++) {
				uint64_t item = items[1 + i];
				if ((item & 15) == me()) {
					// printf("Inserting = %lu\n", item);
					for (size_t k = 0; k < _nb_hash; k++) {
						uint64_t h0 = simplehash16_64(item, k) & _dpu_size_reduced;
						mram_update_byte_atomic(&blooma_starts[me()][h0 >> 3], insert_atomic, bit_mask[h0 & 7]);
					}
				}
			}
			break;
		
		} case LOOKUP: {
			
			nb_positive_lookups[me()] = 0;
			uint64_t nb_items = items[0];
			for (int i = 0; i < nb_items; i++) {
				uint64_t item = items[1 + i];
				if ((item & 15) == me()) {
					bool result = contains(item);
					nb_positive_lookups[me()] += result;
					printf("%d", result);
				}
			}

			barrier_wait(&end_lookup_barrier);
			
			if (me() == 0) {
				printf("\n");
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