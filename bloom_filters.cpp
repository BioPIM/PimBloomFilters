#include <dpu>
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <iostream>
#include <mutex>
#include <omp.h>
#include <thread>
#include <queue>
#include <unordered_map>
#include <unistd.h>

#include "bloom_filters_common.h"

#define NB_THREADS 8

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

    PimBloomFilter(	const int nb_ranks,
					const int size2,
					const uint32_t nb_hash = 4,
					const Implementation implem = BASIC,
					const char* dpu_profile = DpuProfile::HARDWARE)
			: _nb_ranks(nb_ranks), _size2(size2), _nb_hash(nb_hash), _hash_functions(nb_hash) {
		
		if (size2 < 4) {
			throw std::invalid_argument(std::string("Error: bloom size2 must be >= 4"));
		}

		if (nb_ranks <= 0) {
			throw std::invalid_argument(std::string("Error: number of DPUs ranks must be >= 1"));
		}

		if (nb_hash <= 0) {
			throw std::invalid_argument(std::string("Error: number of hash functions must be >= 1"));
		}
		
		_size = (1 << _size2);
		_size_reduced = _size - 1;

		if ((_dpu_size2 >> 3) > MAX_BLOOM_DPU_SIZE2) {
			throw std::invalid_argument(
				  std::string("Error: filter size2 per DPU is bigger than max space available (")
				+ std::to_string(_dpu_size2)
				+ std::string(" > ")
				+ std::to_string(MAX_BLOOM_DPU_SIZE2)
				+ std::string("), try to reduce the size or increase the number of DPUs")
			);
		}

		const char* binary_name = get_dpu_binary_name(implem);
		_sets = (dpu_set_t*) malloc(_nb_ranks * sizeof(dpu_set_t));
	
		// Alloc in parallel
		#pragma omp parallel for num_threads(NB_THREADS)
		for (int r = 0; r < _nb_ranks; r++) {
			DPU_ASSERT(dpu_alloc_ranks(1, dpu_profile, &_sets[r]));
			DPU_ASSERT(dpu_load(_sets[r], binary_name, NULL));
		}

		// This part must be sequential
		_nb_dpu = 0;
		_nb_dpu_per_rank = (int*) malloc((_nb_ranks + 1) * sizeof(int));
		_nb_dpu_per_rank[0] = 0;
		for (int r = 0; r < _nb_ranks; r++) {
			uint32_t nr_dpus;
			DPU_ASSERT(dpu_get_nr_dpus(_sets[r], &nr_dpus));
			_nb_dpu += nr_dpus;
			_nb_dpu_per_rank[r + 1] = _nb_dpu_per_rank[r] + nr_dpus;
		}

		if (_size < _nb_dpu) {
			throw std::invalid_argument(std::string("Error: Asking too little space per DPU, use less DPUs or a bigger filter"));
		}

		_dpu_size2 = ceil(log(_size / (_nb_dpu * 16)) / log(2));

		// Broadcast info and launch in parallel
		uint32_t dpu_indexes[_nb_dpu];
		#pragma omp parallel for num_threads(NB_THREADS)
		for (int r = 0; r < _nb_ranks; r++) {
			DPU_ASSERT(dpu_broadcast_to(_sets[r], "_dpu_size2", 0, &_dpu_size2, sizeof(_dpu_size2), DPU_XFER_DEFAULT));
			DPU_ASSERT(dpu_broadcast_to(_sets[r], "_nb_hash", 0, &_nb_hash, sizeof(_nb_hash), DPU_XFER_DEFAULT));

			uint32_t offset = _nb_dpu_per_rank[r];
			DPU_FOREACH(_sets[r], _dpu, _dpu_idx) {
				dpu_indexes[_dpu_idx + offset] = _dpu_idx + offset;
				DPU_ASSERT(dpu_prepare_xfer(_dpu, &dpu_indexes[_dpu_idx + offset]));
			}
			DPU_ASSERT(dpu_push_xfer(_sets[r], DPU_XFER_TO_DPU, "_dpu_idx", 0, sizeof(uint32_t), DPU_XFER_DEFAULT));

			launch_rank(r, BloomMode::BLOOM_INIT);
		}

		for (int r = 0; r < _nb_ranks; r++) {
			sync_status.push_back(0);
			rank_status.push_back(true);
			sync_mutexes.push_back(new std::mutex());
		}
		for (int r = 0; r < _nb_ranks; r++) {
			std::thread t(&PimBloomFilter::sync_rank, this, r);
			t.detach();
		}

	}

	~PimBloomFilter() {
		#pragma omp parallel for num_threads(NB_THREADS)
		for (int r = 0; r < _nb_ranks; r++) {
			DPU_ASSERT(dpu_free(_sets[r]));
		}
		free(_sets);
		free(_nb_dpu_per_rank);

		for (int r = 0; r < _nb_ranks; r++) {
			delete sync_mutexes[r];
		}
	}

	void insert(std::vector<uint64_t>& items) {

		// uint64_t* buffer = (uint64_t*) malloc(sizeof(uint64_t) * (MAX_NB_ITEMS_PER_DPU + 1));
		// uint32_t* buffer2 = (uint32_t*) malloc(sizeof(uint32_t) * (MAX_NB_ITEMS_PER_DPU + 1));
		// buffer[0] = 0;
		
		// size_t buffer_index = 0;
		// for (int i = 0; i < items.size(); i++) {
		// 	uint64_t item = items[i];
		// 	buffer[0]++;
 		// 	buffer[buffer[0]] = item;

		// 	uint64_t h0 = this->_hash_functions(item, 0);
		// 	uint32_t rank = fastrange32(h0, _nb_ranks);
		// 	int nb_dpus_in_rank = _nb_dpu_per_rank[rank + 1] - _nb_dpu_per_rank[rank];
		// 	uint32_t dpu_idx = _nb_dpu_per_rank[rank] + fastrange32(h0, nb_dpus_in_rank);
		// 	buffer2[buffer[0]] = dpu_idx;

		// 	if ((buffer[0] == MAX_NB_ITEMS_PER_DPU) || (i == (items.size() - 1))) {
		// 		#pragma omp parallel for num_threads(NB_THREADS)
		// 		for (int rank = 0; rank < _nb_ranks; rank++) {
		// 			for (int d = _nb_dpu_per_rank[rank]; d < _nb_dpu_per_rank[rank + 1]; d++) {
		// 				prepare_dpu_launch_async(rank);
		// 				DPU_ASSERT(dpu_broadcast_to(_sets[rank], "items", 0, buffer, sizeof(uint64_t) * CEIL8(MAX_NB_ITEMS_PER_DPU + 1 + 7), DPU_XFER_DEFAULT));
		// 				DPU_ASSERT(dpu_broadcast_to(_sets[rank], "items_keys", 0, buffer2, sizeof(uint32_t) * CEIL8(MAX_NB_ITEMS_PER_DPU + 1 + 7), DPU_XFER_DEFAULT));
		// 				// DPU_FOREACH(_sets[rank], _dpu, _dpu_idx) {
		// 				// 	DPU_ASSERT(dpu_prepare_xfer(_dpu, buffer));
		// 				// }
		// 				// DPU_ASSERT(dpu_push_xfer(_sets[rank], DPU_XFER_TO_DPU, "items", 0, sizeof(uint64_t) * CEIL8(MAX_NB_ITEMS_PER_DPU + 1 + 7), DPU_XFER_DEFAULT));
		// 				launch_rank_async(rank, BloomMode::BLOOM_INSERT);
		// 				break;
		// 			}
		// 		}
		// 		buffer[0] = 0;
		// 	}

		// }



		
		// int item_idx = 0;
		// bool done = false;
		// uint64_t* data = items.data();
		// while (!done) {
		// 	for (int rank = 0; rank < _nb_ranks; rank++) {
		// 		if (item_idx >= items.size()) {
		// 			done = true;
		// 			break;
		// 		} else {
		// 			prepare_dpu_launch_async(rank);
		// 			DPU_FOREACH(_sets[rank], _dpu, _dpu_idx) {
		// 				if (item_idx < items.size()) {
		// 					DPU_ASSERT(dpu_prepare_xfer(_dpu, &data[item_idx]));
		// 					// item_idx += MAX_NB_ITEMS_PER_DPU;
		// 				}
		// 			}
		// 			DPU_ASSERT(dpu_push_xfer(_sets[rank], DPU_XFER_TO_DPU, "items", 0, sizeof(uint64_t) * CEIL8(MAX_NB_ITEMS_PER_DPU), DPU_XFER_DEFAULT));
		// 			launch_rank_async(rank, BloomMode::BLOOM_INSERT);
		// 		}
		// 	}
		// 	item_idx += MAX_NB_ITEMS_PER_DPU;
		// }
		// #pragma omp parallel for num_threads(NB_THREADS)
		// for (int rank = 0; rank < _nb_ranks; rank++) {
		// 	prepare_dpu_launch_async(rank);
		// }

		// std::vector<std::vector<uint64_t*>> t_buckets;
		// t_buckets.reserve(2);
		// for (int t = 0; t < 2; t++) {
		// 	std::vector<uint64_t*> buckets;
		// 	buckets.reserve(_nb_dpu);
		// 	for (int d = 0; d < _nb_dpu; d++) {
		// 		buckets.push_back((uint64_t*) malloc(sizeof(uint64_t) * (MAX_NB_ITEMS_PER_DPU + 1)));
		// 		buckets[d][0] = 0;
		// 	}
		// 	t_buckets.push_back(buckets);
		// }

		// // std::vector<std::mutex*> mutexes;
		// // for (int r = 0; r < _nb_ranks; r++) {
		// // 	mutexes.push_back(new std::mutex());
		// // }

		// uint64_t bucket_size = _size / _nb_dpu;

		// #pragma omp parallel for num_threads(2)
		// for (int t = 0; t < 2; t++) {
		// 	std::vector<uint64_t*> buckets = t_buckets[t];
		// 	for (int i = t; i < items.size(); i += 2) {
		// 		uint64_t item = items[i];
		// 		uint64_t h0 = this->_hash_functions(item, 0);
				
		// 		uint32_t rank = fastrange32(h0, _nb_ranks);
		// 		int nb_dpus_in_rank = _nb_dpu_per_rank[rank + 1] - _nb_dpu_per_rank[rank];
		// 		uint32_t bucket_idx = _nb_dpu_per_rank[rank] + fastrange32(h0, nb_dpus_in_rank);
				
		// 		// mutexes[rank]->lock();
		// 		uint64_t* bucket = buckets[bucket_idx];

		// 		if (bucket[0] == MAX_NB_ITEMS_PER_DPU) {

		// 			if (t == 1) { return; }

		// 			// Launch
		// 			prepare_dpu_launch_async(rank);
		// 			DPU_FOREACH(_sets[rank], _dpu, _dpu_idx) {
		// 				DPU_ASSERT(dpu_prepare_xfer(_dpu, buckets[_nb_dpu_per_rank[rank] + _dpu_idx]));
		// 			}
		// 			DPU_ASSERT(dpu_push_xfer(_sets[rank], DPU_XFER_TO_DPU, "items", 0, sizeof(uint64_t) * CEIL8(MAX_NB_ITEMS_PER_DPU + 8), DPU_XFER_DEFAULT));
		// 			launch_rank_async(rank, BloomMode::BLOOM_INSERT);

		// 			std::thread t1(&PimBloomFilter::sync_rank, this, rank);
		// 			t1.detach();

		// 			// Reset buckets
		// 			for (int d = _nb_dpu_per_rank[rank]; d < _nb_dpu_per_rank[rank + 1]; d++) {
		// 				buckets[d][0] = 0;
		// 			}

		// 		}
		// 		bucket[0]++;
		// 		bucket[bucket[0]] = item;
		// 		// mutexes[rank]->unlock();
		// 	}
		// }

		// // Launch a last round for the ranks that need it
		// for (int t = 0; t < 2; t++) {
		// 	std::vector<uint64_t*> buckets = t_buckets[t];
		// 	#pragma omp parallel for num_threads(NB_THREADS)
		// 	for (int rank = 0; rank < _nb_ranks; rank++) {
		// 		prepare_dpu_launch_async(rank);
		// 		for (int d = _nb_dpu_per_rank[rank]; d < _nb_dpu_per_rank[rank + 1]; d++) {
		// 			uint64_t* bucket = buckets[d];
		// 			if (bucket[0]) {
		// 				DPU_FOREACH(_sets[rank], _dpu, _dpu_idx) {
		// 					DPU_ASSERT(dpu_prepare_xfer(_dpu, buckets[_nb_dpu_per_rank[rank] + _dpu_idx]));
		// 				}
		// 				DPU_ASSERT(dpu_push_xfer(_sets[rank], DPU_XFER_TO_DPU, "items", 0, sizeof(uint64_t) * CEIL8(MAX_NB_ITEMS_PER_DPU + 8), DPU_XFER_DEFAULT));
		// 				launch_rank(rank, BloomMode::BLOOM_INSERT);
		// 				break;
		// 			}
		// 		}
		// 	}
		// }

		// for (int r = 0; r < _nb_ranks; r++) {
		// 	delete mutexes[r];
		// }

		int done = 0;
		std::queue<std::pair<int, std::vector<uint64_t*>*>> ranks_ready;
		std::mutex rank_ready_mutex;
		std::mutex done_mutex;

		omp_set_nested(1);
		#pragma omp parallel sections
		{

			// Filling buffers
			#pragma omp section
			{
				#pragma omp parallel num_threads(5)
				{
					int nb_threads = omp_get_num_threads();
					int tid = omp_get_thread_num();
					std::unordered_map<int, std::vector<uint64_t*>*> rank_buckets = std::unordered_map<int, std::vector<uint64_t*>*>();
					rank_buckets.reserve(_nb_ranks);

					for (int i = tid; i < items.size(); i += nb_threads) {
						uint64_t item = items[i];
						uint64_t h0 = this->_hash_functions(item, 0);
						
						uint32_t rank = fastrange32(h0, _nb_ranks);
						int nb_dpus_in_rank = _nb_dpu_per_rank[rank + 1] - _nb_dpu_per_rank[rank];
						uint32_t bucket_idx = fastrange32(h0, nb_dpus_in_rank);

						if (rank_buckets.find(rank) == rank_buckets.end()) {
							std::vector<uint64_t*>* buckets = new std::vector<uint64_t*>();
							for (int d = 0; d < nb_dpus_in_rank; d++) {
								buckets->push_back((uint64_t*) malloc(sizeof(uint64_t) * (MAX_NB_ITEMS_PER_DPU + 1)));
								(*buckets)[d][0] = 0;
							}
							rank_buckets[rank] = buckets;
						}

						std::vector<uint64_t*>* buckets = rank_buckets[rank];
						uint64_t* bucket = (*buckets)[bucket_idx];
						bucket[0]++;
						bucket[bucket[0]] = item;

						if (bucket[0] == MAX_NB_ITEMS_PER_DPU) {
							rank_ready_mutex.lock();
							ranks_ready.push(std::pair<int, std::vector<uint64_t*>*>(rank, buckets));
							rank_ready_mutex.unlock();
							rank_buckets.erase(rank);
						}

					}
					// Add all remaining to queue
					for (auto info : rank_buckets) {
						int rank = info.first;
						std::vector<uint64_t*>* buckets = info.second;
						rank_ready_mutex.lock();
						ranks_ready.push(std::pair<int, std::vector<uint64_t*>*>(rank, buckets));
						rank_ready_mutex.unlock();
					}
					done_mutex.lock();
					done++; // Must be done **after** adding all remaining to queue!
					done_mutex.unlock();
				}
			}

			// Launching ranks
			#pragma omp section
			{
				while (done < 2 || !ranks_ready.empty()) {
					if (!ranks_ready.empty()) {
						auto info = ranks_ready.front();
						rank_ready_mutex.lock();
						ranks_ready.pop();
						int rank = info.first;
						if (rank_status[rank]) {
							rank_ready_mutex.unlock();
							std::vector<uint64_t*>* buckets = info.second;

							prepare_dpu_launch_async(rank);
							DPU_FOREACH(_sets[rank], _dpu, _dpu_idx) {
								DPU_ASSERT(dpu_prepare_xfer(_dpu, (*buckets)[_dpu_idx]));
							}
							DPU_ASSERT(dpu_push_xfer(_sets[rank], DPU_XFER_TO_DPU, "items", 0, sizeof(uint64_t) * CEIL8(MAX_NB_ITEMS_PER_DPU + 1), DPU_XFER_DEFAULT));
							
							launch_rank_async(rank, BloomMode::BLOOM_INSERT);
							// std::cout << "Launching rank " << rank << std::endl;

							// Cleaning
							for (auto bucket : (*buckets)) {
								free(bucket);
							}
							delete buckets;

						} else {
							ranks_ready.push(info);
							rank_ready_mutex.unlock();
						}
						
					} else {
					}
				}
				// Wait for everything to finish
				for (int rank = 0; rank < _nb_ranks; rank++) {
					prepare_dpu_launch_async(rank);
					sync_mutexes[rank]->lock();
					sync_status[rank] = 2;
					sync_mutexes[rank]->unlock();
				}

			}

		}

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
			int bucket_idx = h0 / bucket_size;
			if (buckets[bucket_idx][0] == MAX_NB_ITEMS_PER_DPU) {

				int rank = get_rank_from_dpu(bucket_idx);

				// Launch
				prepare_dpu_launch_async(rank);
				DPU_FOREACH(_sets[rank], _dpu, _dpu_idx) {
					DPU_ASSERT(dpu_prepare_xfer(_dpu, buckets[_nb_dpu_per_rank[rank] + _dpu_idx]));
				}
				DPU_ASSERT(dpu_push_xfer(_sets[rank], DPU_XFER_TO_DPU, "items", 0, sizeof(uint64_t) * CEIL8(MAX_NB_ITEMS_PER_DPU + 1 + 7), DPU_XFER_DEFAULT));
				launch_rank_async(rank, BloomMode::BLOOM_LOOKUP);

				uint32_t nb_positive_lookups[_nb_dpu_per_rank[rank + 1]];
				
				DPU_FOREACH(_sets[rank], _dpu, _dpu_idx) {
					DPU_ASSERT(dpu_prepare_xfer(_dpu, &nb_positive_lookups[_dpu_idx]));
				}
				DPU_ASSERT(dpu_push_xfer(_sets[rank], DPU_XFER_FROM_DPU, "result", 0, sizeof(uint32_t), DPU_XFER_DEFAULT));

				// Reset buckets
				for (int d = _nb_dpu_per_rank[rank]; d < _nb_dpu_per_rank[rank + 1]; d++) {
					buckets[d][0] = 0;
					total_nb_positive_lookups += nb_positive_lookups[d - _nb_dpu_per_rank[rank]];
				}

			}
			buckets[bucket_idx][0]++;
 			buckets[bucket_idx][buckets[bucket_idx][0]] = item;
		}

		// Launch a last round for the ranks that need it
		for (int rank = 0; rank < _nb_ranks; rank++) {
			for (int d = _nb_dpu_per_rank[rank]; d < _nb_dpu_per_rank[rank + 1]; d++) {
				if (buckets[d][0]) {
					prepare_dpu_launch_async(rank);
					DPU_FOREACH(_sets[rank], _dpu, _dpu_idx) {
						DPU_ASSERT(dpu_prepare_xfer(_dpu, buckets[_nb_dpu_per_rank[rank] + _dpu_idx]));
					}
					DPU_ASSERT(dpu_push_xfer(_sets[rank], DPU_XFER_TO_DPU, "items", 0, sizeof(uint64_t) * CEIL8(MAX_NB_ITEMS_PER_DPU + 1 + 7), DPU_XFER_DEFAULT));
					launch_rank(rank, BloomMode::BLOOM_LOOKUP);
					
					uint32_t nb_positive_lookups[_nb_dpu_per_rank[rank + 1]];
					DPU_FOREACH(_sets[rank], _dpu, _dpu_idx) {
						DPU_ASSERT(dpu_prepare_xfer(_dpu, &nb_positive_lookups[_dpu_idx]));
					}
					DPU_ASSERT(dpu_push_xfer(_sets[rank], DPU_XFER_FROM_DPU, "result", 0, sizeof(uint32_t), DPU_XFER_DEFAULT));
					for (int d = _nb_dpu_per_rank[rank]; d < _nb_dpu_per_rank[rank + 1]; d++) {
						total_nb_positive_lookups += nb_positive_lookups[d - _nb_dpu_per_rank[rank]];
					}
					break;
				}
			}
		}
		return total_nb_positive_lookups; // FIXME
	}

	/// @brief Computes the weight of the filter
	/// @return number of bits set to 1 in the filter
	uint32_t get_weight() {
		
		// Launch in parallel all ranks
		#pragma omp parallel for num_threads(NB_THREADS)
		for (int rank = 0; rank < _nb_ranks; rank++) {
			launch_rank_async(rank, BloomMode::BLOOM_WEIGHT);
		}

		// Reduce results
		uint32_t total_weight = 0;
		for (int rank = 0; rank < _nb_ranks; rank++) {
			prepare_dpu_launch_async(rank);
			uint32_t weights[_nb_dpu_per_rank[rank + 1]];
			DPU_FOREACH(_sets[rank], _dpu, _dpu_idx) {
				DPU_ASSERT(dpu_prepare_xfer(_dpu, &weights[_dpu_idx]));
			}
			DPU_ASSERT(dpu_push_xfer(_sets[rank], DPU_XFER_FROM_DPU, "result", 0, sizeof(uint32_t), DPU_XFER_DEFAULT));
			for (int d = _nb_dpu_per_rank[rank]; d < _nb_dpu_per_rank[rank + 1]; d++) {
				total_weight += weights[d - _nb_dpu_per_rank[rank]];
			}
		}

		return total_weight;

	}

	// Wrappers for single items

	/// @brief Inserts a single item in the filter
	/// @param item item to insert
	void insert (const uint64_t& item) { auto items = std::vector<uint64_t>{item}; insert(items); }

	uint32_t contains (const uint64_t& item)  { return contains(std::vector<uint64_t>{item}); }

	// Misc
	double get_reference_false_positive_probability(const size_t nb_items) {
		return pow(1.0 - exp(-(double) _nb_hash * (double) nb_items / (double) _size), (double) _nb_hash);
	}

private:

	dpu_set_t* _sets;
	int _nb_dpu;
	int* _nb_dpu_per_rank;
	int _nb_ranks;
	struct dpu_set_t _dpu;
	uint32_t _dpu_idx;

	int _size2;
	uint64_t _size;
	uint64_t _size_reduced;
	int _dpu_size2;
	int _nb_hash;

	BloomHashFunctors _hash_functions;

	const char* get_dpu_binary_name(const Implementation implem) {
		switch (implem) {
			case BASIC_CACHE_ITEMS:
				return "bloom_filters_dpu2"; 
			default:
				return "bloom_filters_dpu1";
		}
	}

	void launch_rank(int rank, enum BloomMode mode) {
		DPU_ASSERT(dpu_broadcast_to(_sets[rank], "_mode", 0, &mode, sizeof(mode), DPU_XFER_DEFAULT));
		DPU_ASSERT(dpu_launch(_sets[rank], DPU_SYNCHRONOUS));
		read_dpu_log(rank);
	}

	void prepare_dpu_launch_async(int rank) {
		DPU_ASSERT(dpu_sync(_sets[rank]));
		// read_dpu_log(rank);
	}

	void launch_rank_async(int rank, enum BloomMode mode) {
		DPU_ASSERT(dpu_broadcast_to(_sets[rank], "_mode", 0, &mode, sizeof(mode), DPU_XFER_DEFAULT));
		DPU_ASSERT(dpu_launch(_sets[rank], DPU_ASYNCHRONOUS));
		dpu_callback(_sets[rank], PimBloomFilter::rank_done, new std::pair<PimBloomFilter*, int>(this, rank), DPU_CALLBACK_ASYNC);
		rank_status[rank] = false;
		sync_mutexes[rank]->lock();
		sync_status[rank] = 1;
		sync_mutexes[rank]->unlock();
	}

	void read_dpu_log(int rank) {
		#ifdef LOG_DPU
		DPU_FOREACH(_sets[rank], _dpu) {
			DPU_ASSERT(dpu_log_read(_dpu, stdout));
		}
		#endif
	}

	int get_rank_from_dpu(int dpu_idx) {
		for (int rank = 0; rank < _nb_ranks; rank++) {
			if (dpu_idx < _nb_dpu_per_rank[rank + 1]) {
				return rank;
			}
		}
		return _nb_ranks - 1;
	}

	std::vector<int> sync_status;
	std::vector<bool> rank_status;
	std::mutex rank_mutex;
	std::vector<std::mutex*> sync_mutexes;
	void sync_rank(int rank) {
		while (sync_status[rank] < 2) {
			if (sync_status[rank] == 1) {
				sync_mutexes[rank]->lock();
				sync_status[rank] = 0;
				sync_mutexes[rank]->unlock();
				DPU_ASSERT(dpu_sync(_sets[rank]));
			}
			usleep(100);
		}
	}

	static dpu_error_t rank_done(struct dpu_set_t set, uint32_t rank_id, void* arg) {
		std::pair<PimBloomFilter*, int>* info = (std::pair<PimBloomFilter*, int>*) arg;
		// std::cout << "Rank " << info->second << " is done" << std::endl;
		info->first->rank_mutex.lock();
		info->first->rank_status[info->second] = true;
		info->first->rank_mutex.unlock();
		delete info;
		return DPU_OK;
	}

	static inline uint32_t fastrange32(uint32_t word, uint32_t p) {
		return (uint32_t)(((uint64_t)word * (uint64_t)p) >> 32);
	}

};