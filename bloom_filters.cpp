#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <iostream>
#include <mutex>
#include <omp.h>
#include <queue>

#include "pim_rankset.cpp"
#include "bloom_filters_common.h"

#define NB_THREADS 8

void _worker_done() {}

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
					const char* dpu_profile = DpuProfile::HARDWARE,
					bool print_dpu_logs = false,
               		bool do_trace_debug = false) : _size2(size2), _nb_hash(nb_hash), _hash_functions(nb_hash) {
		
		if (size2 < 4) {
			throw std::invalid_argument(std::string("Error: bloom size2 must be >= 4"));
		}

		if (nb_ranks < 1) {
			throw std::invalid_argument(std::string("Error: number of DPUs ranks must be >= 1"));
		}

		if (nb_hash < 1) {
			throw std::invalid_argument(std::string("Error: number of hash functions must be >= 1"));
		}
		
		_size = (1 << _size2);
		_size_reduced = _size - 1;

		_pim_rankset = new PimRankSet(nb_ranks, NB_THREADS, dpu_profile, get_dpu_binary_name(implem), print_dpu_logs, do_trace_debug);

		if (_size < _pim_rankset->get_nb_dpu()) {
			throw std::invalid_argument(std::string("Error: Asking too little space per DPU, use less DPUs or a bigger filter"));
		}

		_dpu_size2 = ceil(log(_size / (_pim_rankset->get_nb_dpu() * 16)) / log(2));

		if ((_dpu_size2 >> 3) > MAX_BLOOM_DPU_SIZE2) {
			throw std::invalid_argument(
				  std::string("Error: filter size2 per DPU is bigger than max space available (")
				+ std::to_string(_dpu_size2)
				+ std::string(" > ")
				+ std::to_string(MAX_BLOOM_DPU_SIZE2)
				+ std::string("), try to reduce the size or increase the number of DPUs")
			);
		}

		_pim_rankset->for_each_rank([this](int rank_id) {
			int token = _pim_rankset->wait_reserve_rank(rank_id);
			_pim_rankset->broadcast_to_rank(rank_id, "_dpu_size2", 0, &_dpu_size2, sizeof(_dpu_size2));
			_pim_rankset->broadcast_to_rank(rank_id, "_nb_hash", 0, &_nb_hash, sizeof(_nb_hash));
			broadcast_mode(rank_id, BloomMode::BLOOM_INIT);
			_pim_rankset->launch_rank_sync(rank_id, token);
		}, true);

	}

	~PimBloomFilter() {
		delete _pim_rankset;
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
		int* done_addr = &done; // Using address because weird behavior with nested omp otherwise, change is not seen in launchers
		std::mutex done_mutex;

		std::queue<std::pair<int, std::vector<uint64_t*>*>> ranks_ready;
		std::mutex rank_ready_mutex;

		int nb_items = items.size();
		int nb_ranks = _pim_rankset->get_nb_ranks();
		int nb_workers = 5;

		omp_set_nested(1);
		#pragma omp parallel sections
		{			

			// Workers: fill buckets
			#pragma omp section
			{
				#pragma omp parallel num_threads(nb_workers)
				{
					int tid = omp_get_thread_num();
					std::vector<std::vector<uint64_t*>*> rank_buckets = std::vector<std::vector<uint64_t*>*>(nb_ranks, NULL);
					
					for (int i = tid; i < nb_items; i += nb_workers) {
						uint64_t item = items[i];
						auto dispatch_data = _get_rank_dpu_id_from_item(item);
						uint32_t rank_id = dispatch_data.first;
						uint32_t bucket_idx = dispatch_data.second;
						int nb_dpus_in_rank = _pim_rankset->get_nb_dpu_in_rank(rank_id);

						std::vector<uint64_t*>* buckets = rank_buckets[rank_id];

						if (buckets == NULL) {
							buckets = new std::vector<uint64_t*>(nb_dpus_in_rank, NULL);
							for (int d = 0; d < nb_dpus_in_rank; d++) {
								(*buckets)[d] = ((uint64_t*) malloc(sizeof(uint64_t) * (MAX_NB_ITEMS_PER_DPU + 1)));
								(*buckets)[d][0] = 0;
							}
							rank_buckets[rank_id] = buckets;
						}
						
						uint64_t* bucket = (*buckets)[bucket_idx];
						bucket[0]++;
						bucket[bucket[0]] = item;

						if (bucket[0] == MAX_NB_ITEMS_PER_DPU) {
							rank_ready_mutex.lock();
							ranks_ready.push(std::pair<int, std::vector<uint64_t*>*>(rank_id, buckets));
							rank_ready_mutex.unlock();
							rank_buckets[rank_id] = NULL;
						}

					}
					// Add all remaining to queue
					for (int rank_id = 0; rank_id < nb_ranks; rank_id++) {
						std::vector<uint64_t*>* buckets = rank_buckets[rank_id];
						if (buckets != NULL) {
							rank_ready_mutex.lock();
							ranks_ready.push(std::pair<int, std::vector<uint64_t*>*>(rank_id, buckets));
							rank_ready_mutex.unlock();
						}
					}
					done_mutex.lock();
					(*done_addr)++; // Must be done **after** adding all remaining to queue!
					done_mutex.unlock();
					_worker_done(); // Call for trace to see when workers are done
				}
			}

			// Launcher
			#pragma omp section
			{
				#pragma omp parallel num_threads(1)
				{
					int token;
					int items_done = 0, rounds_launched = 0;
					while (true) {
						if (ranks_ready.empty()) {
							if ((*done_addr) >= nb_workers) {
								break;
							}
						} else {
							rank_ready_mutex.lock();
							auto info = ranks_ready.front();
							int rank_id = info.first;
							if ((token = _pim_rankset->try_reserve_rank(rank_id)) != PimRankSet::CANNOT_RESERVE) {

								ranks_ready.pop();
								rank_ready_mutex.unlock();
								
								std::vector<uint64_t*>* buckets = info.second;
								_pim_rankset->send_data_to_rank<uint64_t>(rank_id, "items", 0, *buckets, sizeof(uint64_t) * CEIL8(MAX_NB_ITEMS_PER_DPU + 1));
								broadcast_mode(rank_id, BloomMode::BLOOM_INSERT);
								_pim_rankset->launch_rank_async(rank_id, token);
								rounds_launched++;
								// std::cout << "Launching rank " << rank_id << " with " << std::endl;
								for (auto bucket : (*buckets)) {
									items_done += bucket[0];
									// std::cout << bucket[0] << " ";
								}
								// std::cout << std::endl;

								// Cleaning
								for (auto bucket : (*buckets)) {
									free(bucket);
								}
								delete buckets;

							} else {
								if (ranks_ready.size() > 1) { // No need to pop front and push back if nothing else in queue
									ranks_ready.pop();
									ranks_ready.push(info);
								}
								rank_ready_mutex.unlock();
							}
						}
					}
					std::cout << items_done << " items done" << std::endl;
					std::cout << rounds_launched << " rounds launched" << std::endl;
				}
				// Wait for DPUs to finish
				_pim_rankset->for_each_rank([this](int rank_id) {
					_pim_rankset->wait_rank_done(rank_id);
				}, true);

			}

		}

		// int nb_dpu = _pim_rankset->get_nb_dpu();
		// int approximate_load = items.size() / nb_dpu;
		// std::vector<std::vector<uint64_t>> buckets = std::vector<std::vector<uint64_t>>(nb_dpu, std::vector<uint64_t>());
		// for (auto bucket : buckets) {
		// 	bucket.reserve(approximate_load);
		// }
		// for (auto item : items) {
		// 	auto dispatch_data = _get_rank_dpu_id_from_item(item);
		// 	uint32_t rank_id = dispatch_data.first;
		// 	uint32_t bucket_idx = dispatch_data.second;
		// 	int start_idx = _pim_rankset->get_cum_dpu_idx_for_rank(rank_id);
		// 	// buckets[start_idx + bucket_idx].push_back(item);
		// }

	}

	uint32_t contains(const std::vector<uint64_t>& items) {

		uint32_t total_nb_positive_lookups = 0;
		// int nb_dpu = _pim_rankset->get_nb_dpu();

		// std::vector<uint64_t*> buckets;
		// for (size_t d = 0; d < nb_dpu; d++) {
		// 	buckets.push_back((uint64_t*) malloc(sizeof(uint64_t) * (MAX_NB_ITEMS_PER_DPU + 1)));
		// 	buckets[d][0] = 0;
		// }

		// for (auto item : items) {
		// 	auto dispatch_data = _get_rank_dpu_id_from_item(item);
		// 	uint32_t rank_id = dispatch_data.first;
		// 	uint32_t bucket_idx = dispatch_data.second;
		// 	if (buckets[bucket_idx][0] == MAX_NB_ITEMS_PER_DPU) {

		// 		int start_idx = _pim_rankset->get_cum_dpu_idx_for_rank(rank_id);
		// 		int nb_dpu_in_rank = _pim_rankset->get_nb_dpu_in_rank(rank_id);

		// 		// Launch
		// 		_pim_rankset->wait_rank_ready(rank_id);
		// 		_pim_rankset->send_data_to_rank<uint64_t>(rank_id, "items", 0,
		// 			std::vector<uint64_t*>(buckets.begin() + start_idx, buckets.begin() + start_idx + nb_dpu_in_rank),
		// 			sizeof(uint64_t) * CEIL8(MAX_NB_ITEMS_PER_DPU + 1));
		// 		broadcast_mode(rank_id, BloomMode::BLOOM_LOOKUP);
		// 		_pim_rankset->launch_rank_sync(rank_id);

		// 		total_nb_positive_lookups += _pim_rankset->get_reduced_sum_from_rank<uint32_t>(rank_id, "result", 0, sizeof(uint32_t));

		// 		// Reset buckets
		// 		for (int d = start_idx; d < (start_idx + nb_dpu_in_rank); d++) {
		// 			buckets[d][0] = 0;
		// 		}

		// 	}
		// 	buckets[bucket_idx][0]++;
 		// 	buckets[bucket_idx][buckets[bucket_idx][0]] = item;
		// }

		// // Launch a last round for the ranks that need it
		// for (int rank_id = 0; rank_id < _pim_rankset->get_nb_ranks(); rank_id++) {
		// 	int start_idx = _pim_rankset->get_cum_dpu_idx_for_rank(rank_id);
		// 	int nb_dpu_in_rank = _pim_rankset->get_nb_dpu_in_rank(rank_id);
		// 	for (int d = start_idx; d < (start_idx + nb_dpu_in_rank); d++) {
		// 		if (buckets[d][0]) {
		// 			_pim_rankset->wait_rank_ready(rank_id);
		// 			_pim_rankset->send_data_to_rank<uint64_t>(rank_id, "items", 0,
		// 				std::vector<uint64_t*>(buckets.begin() + start_idx, buckets.begin() + start_idx + nb_dpu_in_rank),
		// 				sizeof(uint64_t) * CEIL8(MAX_NB_ITEMS_PER_DPU + 1));
		// 			broadcast_mode(rank_id, BloomMode::BLOOM_LOOKUP);
		// 			_pim_rankset->launch_rank_sync(rank_id);
		// 			total_nb_positive_lookups += _pim_rankset->get_reduced_sum_from_rank<uint32_t>(rank_id, "result", 0, sizeof(uint32_t));
		// 		}
		// 	}
		// }
		return total_nb_positive_lookups;
	}

	/// @brief Computes the weight of the filter
	/// @return number of bits set to 1 in the filter
	uint32_t get_weight() {

		// Launch in parallel all ranks
		_pim_rankset->for_each_rank([this](int rank_id) {
			int token = _pim_rankset->wait_reserve_rank(rank_id);
			broadcast_mode(rank_id, BloomMode::BLOOM_WEIGHT);
			_pim_rankset->launch_rank_async(rank_id, token);
		}, true);

		// Reduce results in sequential
		uint32_t weight = 0;
		_pim_rankset->for_each_rank([this, &weight](int rank_id) {
			_pim_rankset->wait_rank_done(rank_id);
			weight += _pim_rankset->get_reduced_sum_from_rank<uint32_t>(rank_id, "result", 0, sizeof(uint32_t));
		}, false);

		return weight;

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

	PimRankSet* _pim_rankset;
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

	void broadcast_mode(int rank_id, BloomMode mode) {
		_pim_rankset->broadcast_to_rank(rank_id, "_mode", 0, &mode, sizeof(mode));
	}

	static inline uint32_t fastrange32(uint32_t word, uint32_t p) {
		return (uint32_t)(((uint64_t)word * (uint64_t)p) >> 32);
	}

	std::pair<uint32_t, uint32_t> _get_rank_dpu_id_from_item(uint64_t item) {
		uint64_t h0 = this->_hash_functions(item, 0);
		uint32_t rank_id = fastrange32(h0, _pim_rankset->get_nb_ranks());
		int nb_dpus_in_rank = _pim_rankset->get_nb_dpu_in_rank(rank_id);
		uint64_t h1 = this->_hash_functions(item, 1); // IMPORTANT: use a different hash otherwise some correlation can happen and some indexes may never be picked
		uint32_t dpu_id = fastrange32(h1, nb_dpus_in_rank);
		return std::pair<uint32_t, uint32_t>(rank_id, dpu_id);
	}

};