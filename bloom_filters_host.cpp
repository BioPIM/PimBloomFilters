#include <dpu>
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <vector>
#include <iostream>
#include <random>
#include <algorithm>
#include <omp.h>

#include "bloom_filters_common.h"

#define USE_DPU_SIMULATOR

#ifdef USE_DPU_SIMULATOR
	#define DPU_PROFILE "backend=simulator"
#else
	#define DPU_PROFILE "backend=hw"
#endif

using namespace std;
using namespace dpu;

#define NB_DPU 1
#define NB_THREADS 8

#define NB_ITEMS (1 << 10)
#define NB_NO_ITEMS 500
#define NB_HASH 8
#define BLOOM_SIZE2 11

#define BLOOM_SIZE (1 << BLOOM_SIZE2)

#define NB_REPEATS 1

class BloomHashFunctors {
public:

    BloomHashFunctors(size_t nb_functions, u_int64_t seed = 0) : _nb_functions(nb_functions), user_seed(seed) {
        generate_hash_seed();
    }

    u_int64_t operator()(const u_int64_t& key, size_t idx) {
        return hash64(key, seed_tab[idx]);
    }

private:

	static u_int64_t hash64(u_int64_t key, u_int64_t seed) {
        u_int64_t hash = seed;
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
        static const u_int64_t rbase[NSEEDSBLOOM] = {
            0xAAAAAAAA55555555ULL,  0x33333333CCCCCCCCULL,  0x6666666699999999ULL,  0xB5B5B5B54B4B4B4BULL,
            0xAA55AA5555335533ULL,  0x33CC33CCCC66CC66ULL,  0x6699669999B599B5ULL,  0xB54BB54B4BAA4BAAULL,
            0xAA33AA3355CC55CCULL,  0x33663366CC99CC99ULL
        };

        for (size_t i = 0; i < NSEEDSBLOOM; ++i) { seed_tab[i] = rbase[i]; }
        for (size_t i = 0; i < NSEEDSBLOOM; ++i) { seed_tab[i] = seed_tab[i] * seed_tab[(i + 3) % NSEEDSBLOOM] + user_seed; }
    }

    size_t _nb_functions;
    static const size_t NSEEDSBLOOM = 10;
    u_int64_t seed_tab[NSEEDSBLOOM];
    u_int64_t user_seed;
};

class PimBloomFilter {
public:

	enum Implementation {
		BASIC,
	};

    PimBloomFilter(size_t nb_dpu, size_t size2, uint32_t nb_hash = 4, Implementation implem = BASIC) : _nb_dpu(nb_dpu), _size2(size2), _nb_hash(nb_hash), _hash_functions(nb_hash) {
		
		_size = (1 << _size2);
		_size_reduced = _size - 1;
		_dpu_size2 = ceil(log(_size / NB_DPU) / log(2));
		
		if (_dpu_size2 > MAX_BLOOM_DPU_SIZE2) {
			throw invalid_argument(
				  std::string("Error: filter size2 per DPU is bigger than max space available (")
				+ std::to_string(_dpu_size2)
				+ std::string(" > ")
				+ std::to_string(MAX_BLOOM_DPU_SIZE2)
				+ std::string("), try to reduce the size or increase the number of DPUs")
			);
		}

		DPU_ASSERT(dpu_alloc(nb_dpu, DPU_PROFILE, &_set));
		DPU_ASSERT(dpu_load(_set, get_dpu_binary_name(implem), NULL));

		DPU_ASSERT(dpu_broadcast_to(_set, "_dpu_size2", 0, &_dpu_size2, sizeof(_dpu_size2), DPU_XFER_DEFAULT));
		DPU_ASSERT(dpu_broadcast_to(_set, "_nb_hash", 0, &_nb_hash, sizeof(_nb_hash), DPU_XFER_DEFAULT));
		
		launch_dpu(INIT);
		read_dpu_log();
		
	}

	~PimBloomFilter() {
		DPU_ASSERT(dpu_free(_set));
	}

	void insert(std::vector<u_int64_t> items) {

		std::vector<u_int64_t*> buckets;
		for (size_t d = 0; d < _nb_dpu; d++) {
			buckets.push_back((u_int64_t*) malloc(sizeof(u_int64_t) * (MAX_NB_ITEMS_PER_DPU + 1)));
			buckets[d][0] = 0;
		}

		u_int64_t bucket_size = _size / _nb_dpu;

		for (auto item : items) {
			u_int64_t h0 = this->_hash_functions(item, 0) & _size_reduced;
			size_t bucket_idx = h0 / bucket_size;
			if (buckets[bucket_idx][0] == MAX_NB_ITEMS_PER_DPU) {

				// Launch
				DPU_FOREACH(_set, _dpu, _dpu_idx) {
					DPU_ASSERT(dpu_prepare_xfer(_dpu, buckets[_dpu_idx]));
				}
				DPU_ASSERT(dpu_push_xfer(_set, DPU_XFER_TO_DPU, "items", 0, sizeof(uint64_t) * ((MAX_NB_ITEMS_PER_DPU + 1 + 7) >> 3) << 3, DPU_XFER_DEFAULT));
				launch_dpu(INSERT);
				read_dpu_log();

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
		launch_dpu(INSERT);
		read_dpu_log();
	}

	uint32_t contains(std::vector<u_int64_t> items) {

		uint32_t total_nb_positive_lookups = 0;

		std::vector<u_int64_t*> buckets;
		for (size_t d = 0; d < _nb_dpu; d++) {
			buckets.push_back((u_int64_t*) malloc(sizeof(u_int64_t) * (MAX_NB_ITEMS_PER_DPU + 1)));
			buckets[d][0] = 0;
		}

		u_int64_t bucket_size = _size / _nb_dpu;

		for (auto item : items) {
			u_int64_t h0 = this->_hash_functions(item, 0) & _size_reduced;
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

    uint32_t contains (const u_int64_t& item) {
		std::vector<u_int64_t> items;
		items.push_back(item);
		return contains(items); // FIXME
    }

	double get_reference_false_positive_probability(size_t nb_items) {
		return pow(1.0 - exp(-(double) _nb_hash * (double) nb_items / (double) _size), (double) _nb_hash);
	}

private:

	dpu_set_t _set;
	size_t _nb_dpu;
	struct dpu_set_t _dpu;
	uint32_t _dpu_idx;
    BloomHashFunctors _hash_functions;
	size_t _size2;
	u_int64_t _size;
	u_int64_t _size_reduced;
	size_t _dpu_size2;
	size_t _nb_hash;

	const char* get_dpu_binary_name(Implementation implem) {
		switch (implem) {
			default:
				return "bloom_filters_dpu1";
		}
	}

	void launch_dpu(enum BloomMode mode) {
		DPU_ASSERT(dpu_broadcast_to(_set, "_mode", 0, &mode, sizeof(mode), DPU_XFER_DEFAULT));
		DPU_ASSERT(dpu_launch(_set, DPU_SYNCHRONOUS));
	}

	void read_dpu_log() {
		DPU_FOREACH(_set, _dpu) {
			DPU_ASSERT(dpu_log_read(_dpu, stdout));
		}
	}

	uint32_t launch_lookups(std::vector<u_int64_t*>& buckets) {
		DPU_FOREACH(_set, _dpu, _dpu_idx) {
			DPU_ASSERT(dpu_prepare_xfer(_dpu, buckets[_dpu_idx]));
		}
		DPU_ASSERT(dpu_push_xfer(_set, DPU_XFER_TO_DPU, "items", 0, sizeof(uint64_t) * ((MAX_NB_ITEMS_PER_DPU + 1 + 7) >> 3) << 3, DPU_XFER_DEFAULT));
		launch_dpu(LOOKUP);
		read_dpu_log();

		uint32_t total_nb_positive_lookups = 0, nb_positive_lookups[_nb_dpu];
		DPU_FOREACH(_set, _dpu, _dpu_idx) {
			DPU_ASSERT(dpu_prepare_xfer(_dpu, &nb_positive_lookups[_dpu_idx]));
		}
		DPU_ASSERT(dpu_push_xfer(_set, DPU_XFER_FROM_DPU, "total_nb_positive_lookups", 0, sizeof(uint32_t), DPU_XFER_DEFAULT));
		for (size_t d = 0; d < _nb_dpu; d++) {
			total_nb_positive_lookups += nb_positive_lookups[d];
		}
		return total_nb_positive_lookups;
	}

};

int main() {

	clock_t start = clock();

	// Prepare items
	std::vector<u_int64_t> items(NB_ITEMS);
	for (int i = 0; i < NB_ITEMS; i++) {
		items[i] = i;
	}

	std::vector<u_int64_t> no_items(NB_NO_ITEMS);
	for (int i = 0; i < NB_NO_ITEMS; i++) {
		no_items[i] = i + NB_ITEMS;
	}

	try {
		// Testing lookup results
		uint32_t nb_positive_lookups;
		PimBloomFilter *bloom_filter = new PimBloomFilter(NB_DPU, BLOOM_SIZE2, NB_HASH);
		bloom_filter->insert(items);

		auto rng = std::default_random_engine {};
		std::shuffle(std::begin(items), std::end(items), rng);

		nb_positive_lookups = bloom_filter->contains(items);
		if (nb_positive_lookups == items.size()) {
			cout << "[OK] All items inserted give a positive" << endl;
		} else {
			cout << "[WARNING] There is " << items.size() - nb_positive_lookups << " false negative(s)" << endl;
		}

		nb_positive_lookups = bloom_filter->contains(no_items);
		cout << "False positive frequency is " << (double) nb_positive_lookups / no_items.size() << endl;
		cout << "Reference false positive probability is " << bloom_filter->get_reference_false_positive_probability(items.size()) << endl;

		delete bloom_filter;
	} catch (invalid_argument& e) {
        cerr << e.what() << endl;
        return -1;
    }

	

	clock_t end = clock();
	printf("Host elapsed time: %.2e secs.\n", (double)(end - start) / CLOCKS_PER_SEC);


	// for (int repeat = 0; repeat < NB_REPEATS; repeat++) {}
	// uint32_t each_dpu, dpu_nb_cycles[NB_DPU], dpu_clocks_per_sec[NB_DPU];
	// DPU_FOREACH(set, dpu, each_dpu) {
	// 	DPU_ASSERT(dpu_prepare_xfer(dpu, &dpu_nb_cycles[each_dpu]));
	// }
	// DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_FROM_DPU, "nb_cycles", 0, sizeof(uint32_t), DPU_XFER_DEFAULT));

	// DPU_FOREACH(set, dpu, each_dpu) {
	// 	DPU_ASSERT(dpu_prepare_xfer(dpu, &dpu_clocks_per_sec[each_dpu]));
	// }
	// DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_FROM_DPU, "CLOCKS_PER_SEC", 0, sizeof(uint32_t), DPU_XFER_DEFAULT));
	
	// for (int k = 0; k < NB_DPU; k++) {
	// 	total_sum += dpu_sums[k];
	// 	printf("DPU #%d sum: %u\n", k, dpu_sums[k]);
	// 	printf("DPU #%d cycles: %u\n", k, dpu_nb_cycles[k]);
	// 	printf("DPU #%d time: %.2e secs.\n", k, (double)dpu_nb_cycles[k] / dpu_clocks_per_sec[k]);
	// }
	
	return 0;
}
