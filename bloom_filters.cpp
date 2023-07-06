#include <dpu>
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <iostream>
#include <omp.h>

#include "bloom_filters_common.h"

class BloomHashFunctors {
public:

    BloomHashFunctors(size_t nb_functions, uint64_t seed = 0) : _nb_functions(nb_functions), user_seed(seed) {
        generate_hash_seed();
    }

    uint64_t operator()(const uint64_t& key, size_t idx) {
        return hash64(key, seed_tab[idx]);
    }

private:

	static uint64_t hash64(uint64_t key, uint64_t seed) {
        uint64_t hash = seed;
        hash ^= (hash <<  7) ^  key * (hash >> 3) ^ (~((hash << 11) + (key ^ (hash >> 5))));
        hash = (~hash) + (hash << 21); // hash = (hash << 21) - hash - 1;
        hash = hash ^ (hash >> 24);
        hash = (hash + (hash << 3)) + (hash << 8); // hash * 265
        hash = hash ^ (hash >> 14);
        hash = (hash + (hash << 2)) + (hash << 4); // hash * 21
        hash = hash ^ (hash >> 28);
        hash = hash + (hash << 31);
        return hash;
    }

    void generate_hash_seed() {
        static const uint64_t rbase[NSEEDSBLOOM] = {
            0xAAAAAAAA55555555ULL,  0x33333333CCCCCCCCULL,  0x6666666699999999ULL,  0xB5B5B5B54B4B4B4BULL,
            0xAA55AA5555335533ULL,  0x33CC33CCCC66CC66ULL,  0x6699669999B599B5ULL,  0xB54BB54B4BAA4BAAULL,
            0xAA33AA3355CC55CCULL,  0x33663366CC99CC99ULL
        };

        for (size_t i = 0; i < NSEEDSBLOOM; ++i) { seed_tab[i] = rbase[i]; }
        for (size_t i = 0; i < NSEEDSBLOOM; ++i) { seed_tab[i] = seed_tab[i] * seed_tab[(i + 3) % NSEEDSBLOOM] + user_seed; }
    }

    size_t _nb_functions;
    static const size_t NSEEDSBLOOM = 10;
    uint64_t seed_tab[NSEEDSBLOOM];
    uint64_t user_seed;
};

class PimBloomFilter {
public:

	enum Implementation {
		BASIC,
		BASIC_CACHE_ITEMS,
	};

    PimBloomFilter(	const size_t nb_dpu,
					const size_t size2,
					const uint32_t nb_hash = 4,
					const Implementation implem = BASIC,
					const char* dpu_profile = DpuProfile::HARDWARE)
			: _nb_dpu(nb_dpu), _size2(size2), _nb_hash(nb_hash), _hash_functions(nb_hash) {
		
		if (size2 < 4) {
			throw std::invalid_argument(std::string("Error: bloom size2 must be >= 4"));
		}

		if (nb_dpu <= 0) {
			throw std::invalid_argument(std::string("Error: number of DPUs must be >= 1"));
		}

		if (nb_hash <= 0) {
			throw std::invalid_argument(std::string("Error: number of hash functions must be >= 1"));
		}
		
		_size = (1 << _size2);
		_size_reduced = _size - 1;
		_dpu_size2 = ceil(log(_size / (_nb_dpu * 16)) / log(2));

		if ((_dpu_size2 >> 3) > MAX_BLOOM_DPU_SIZE2) {
			throw std::invalid_argument(
				  std::string("Error: filter size2 per DPU is bigger than max space available (")
				+ std::to_string(_dpu_size2)
				+ std::string(" > ")
				+ std::to_string(MAX_BLOOM_DPU_SIZE2)
				+ std::string("), try to reduce the size or increase the number of DPUs")
			);
		}

		DPU_ASSERT(dpu_alloc(nb_dpu, dpu_profile, &_set));
		DPU_ASSERT(dpu_load(_set, get_dpu_binary_name(implem), NULL));

		DPU_ASSERT(dpu_broadcast_to(_set, "_dpu_size2", 0, &_dpu_size2, sizeof(_dpu_size2), DPU_XFER_DEFAULT));
		DPU_ASSERT(dpu_broadcast_to(_set, "_nb_hash", 0, &_nb_hash, sizeof(_nb_hash), DPU_XFER_DEFAULT));
		
		launch_dpu(BloomMode::BLOOM_INIT);
		
	}

	~PimBloomFilter() {
		DPU_ASSERT(dpu_free(_set));
	}

	void insert(const std::vector<uint64_t>& items) {

		std::vector<uint64_t*> buckets;
		for (size_t d = 0; d < _nb_dpu; d++) {
			buckets.push_back((uint64_t*) malloc(sizeof(uint64_t) * (MAX_NB_ITEMS_PER_DPU + 1)));
			buckets[d][0] = 0;
		}

		uint64_t bucket_size = _size / _nb_dpu;

		for (auto item : items) {
			uint64_t h0 = this->_hash_functions(item, 0) & _size_reduced;
			size_t bucket_idx = h0 / bucket_size;
			if (buckets[bucket_idx][0] == MAX_NB_ITEMS_PER_DPU) {

				// Launch
				prepare_dpu_launch_async();
				DPU_FOREACH(_set, _dpu, _dpu_idx) {
					DPU_ASSERT(dpu_prepare_xfer(_dpu, buckets[_dpu_idx]));
				}
				DPU_ASSERT(dpu_push_xfer(_set, DPU_XFER_TO_DPU, "items", 0, sizeof(uint64_t) * ((MAX_NB_ITEMS_PER_DPU + 1 + 7) >> 3) << 3, DPU_XFER_DEFAULT));
				launch_dpu_async(BloomMode::BLOOM_INSERT);

				// Reset buckets
				for (size_t d = 0; d < _nb_dpu; d++) {
					buckets[d][0] = 0;
				}

			}
			buckets[bucket_idx][0]++;
 			buckets[bucket_idx][buckets[bucket_idx][0]] = item;
		}

		// Last round
		DPU_FOREACH(_set, _dpu, _dpu_idx) {
			DPU_ASSERT(dpu_prepare_xfer(_dpu, buckets[_dpu_idx]));
		}
		DPU_ASSERT(dpu_push_xfer(_set, DPU_XFER_TO_DPU, "items", 0, sizeof(uint64_t) * ((MAX_NB_ITEMS_PER_DPU + 1 + 7) >> 3) << 3, DPU_XFER_DEFAULT));
		launch_dpu(BloomMode::BLOOM_INSERT);
	}

	void insert (const uint64_t& item) {
		std::vector<uint64_t> items;
		items.push_back(item);
		insert(items);
    }

	uint32_t contains(const std::vector<uint64_t>& items) {

		uint32_t total_nb_positive_lookups = 0;

		std::vector<uint64_t*> buckets;
		for (size_t d = 0; d < _nb_dpu; d++) {
			buckets.push_back((uint64_t*) malloc(sizeof(uint64_t) * (MAX_NB_ITEMS_PER_DPU + 1)));
			buckets[d][0] = 0;
		}

		uint64_t bucket_size = _size / _nb_dpu;

		for (auto item : items) {
			uint64_t h0 = this->_hash_functions(item, 0) & _size_reduced;
			size_t bucket_idx = h0 / bucket_size;
			if (buckets[bucket_idx][0] == MAX_NB_ITEMS_PER_DPU) {

				total_nb_positive_lookups += launch_lookups(buckets);

				// Reset buckets
				for (size_t d = 0; d < _nb_dpu; d++) {
					buckets[d][0] = 0;
				}

			}
			buckets[bucket_idx][0]++;
 			buckets[bucket_idx][buckets[bucket_idx][0]] = item;
		}

		// Last round
		total_nb_positive_lookups += launch_lookups(buckets);

		return total_nb_positive_lookups; // FIXME
	}

    uint32_t contains (const uint64_t& item) {
		std::vector<uint64_t> items;
		items.push_back(item);
		return contains(items); // FIXME
    }

	uint32_t get_weight() {
		launch_dpu(BloomMode::BLOOM_WEIGHT);
		uint32_t total_weight = 0, weights[_nb_dpu];
		DPU_FOREACH(_set, _dpu, _dpu_idx) {
			DPU_ASSERT(dpu_prepare_xfer(_dpu, &weights[_dpu_idx]));
		}
		DPU_ASSERT(dpu_push_xfer(_set, DPU_XFER_FROM_DPU, "result", 0, sizeof(uint32_t), DPU_XFER_DEFAULT));
		for (size_t d = 0; d < _nb_dpu; d++) {
			total_weight += weights[d];
		}
		return total_weight;
	}

	double get_reference_false_positive_probability(const size_t nb_items) {
		return pow(1.0 - exp(-(double) _nb_hash * (double) nb_items / (double) _size), (double) _nb_hash);
	}

private:

	dpu_set_t _set;
	size_t _nb_dpu;
	struct dpu_set_t _dpu;
	uint32_t _dpu_idx;
    BloomHashFunctors _hash_functions;
	size_t _size2;
	uint64_t _size;
	uint64_t _size_reduced;
	size_t _dpu_size2;
	size_t _nb_hash;

	const char* get_dpu_binary_name(const Implementation implem) {
		switch (implem) {
			case BASIC_CACHE_ITEMS:
				return "bloom_filters_dpu2"; 
			default:
				return "bloom_filters_dpu1";
		}
	}

	void launch_dpu(enum BloomMode mode) {
		DPU_ASSERT(dpu_broadcast_to(_set, "_mode", 0, &mode, sizeof(mode), DPU_XFER_DEFAULT));
		DPU_ASSERT(dpu_launch(_set, DPU_SYNCHRONOUS));
		read_dpu_log();
	}

	void prepare_dpu_launch_async() {
		DPU_ASSERT(dpu_sync(_set));
		read_dpu_log();
	}

	void launch_dpu_async(enum BloomMode mode) {
		DPU_ASSERT(dpu_broadcast_to(_set, "_mode", 0, &mode, sizeof(mode), DPU_XFER_DEFAULT));
		DPU_ASSERT(dpu_launch(_set, DPU_ASYNCHRONOUS));
	}

	void read_dpu_log() {
		#ifdef LOG_DPU
		DPU_FOREACH(_set, _dpu) {
			DPU_ASSERT(dpu_log_read(_dpu, stdout));
		}
		#endif
	}

	uint32_t launch_lookups(std::vector<uint64_t*>& buckets) {
		DPU_FOREACH(_set, _dpu, _dpu_idx) {
			DPU_ASSERT(dpu_prepare_xfer(_dpu, buckets[_dpu_idx]));
		}
		DPU_ASSERT(dpu_push_xfer(_set, DPU_XFER_TO_DPU, "items", 0, sizeof(uint64_t) * ((MAX_NB_ITEMS_PER_DPU + 1 + 7) >> 3) << 3, DPU_XFER_DEFAULT));
		launch_dpu(BloomMode::BLOOM_LOOKUP);

		uint32_t total_nb_positive_lookups = 0, nb_positive_lookups[_nb_dpu];
		DPU_FOREACH(_set, _dpu, _dpu_idx) {
			DPU_ASSERT(dpu_prepare_xfer(_dpu, &nb_positive_lookups[_dpu_idx]));
		}
		DPU_ASSERT(dpu_push_xfer(_set, DPU_XFER_FROM_DPU, "result", 0, sizeof(uint32_t), DPU_XFER_DEFAULT));
		for (size_t d = 0; d < _nb_dpu; d++) {
			total_nb_positive_lookups += nb_positive_lookups[d];
		}
		return total_nb_positive_lookups;
	}

};