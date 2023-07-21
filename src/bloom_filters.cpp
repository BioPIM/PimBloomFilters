#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <iostream>
#include <sstream>
#include <mutex>
#include <omp.h>
#include <queue>
#include <thread>

#include "spdlog/spdlog.h"

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

    PimBloomFilter(	const int nb_ranks,
					const int size2,
					const uint32_t nb_hash = 4,
					const char* dpu_profile = DpuProfile::HARDWARE) :
						_pim_rankset(PimRankSet(nb_ranks, NB_THREADS, dpu_profile, get_dpu_binary_name().c_str())),
						_size2(size2), _nb_hash(nb_hash), _hash_functions(nb_hash) {
		
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

		if (_size < _pim_rankset.get_nb_dpu()) {
			throw std::invalid_argument(std::string("Error: Asking too little space per DPU, use less DPUs or a bigger filter"));
		}

		_dpu_size2 = ceil(log(_size / (_pim_rankset.get_nb_dpu() * 16)) / log(2));

		if ((_dpu_size2 >> 3) > MAX_BLOOM_DPU_SIZE2) {
			throw std::invalid_argument(
				  std::string("Error: filter size2 per DPU is bigger than max space available (")
				+ std::to_string(_dpu_size2)
				+ std::string(" > ")
				+ std::to_string(MAX_BLOOM_DPU_SIZE2)
				+ std::string("), try to reduce the size or increase the number of DPUs")
			);
		}

		_pim_rankset.for_each_rank([this](size_t rank_id) {
			_pim_rankset.broadcast_to_rank_sync(rank_id, "_dpu_size2", 0, &_dpu_size2, sizeof(_dpu_size2));
			_pim_rankset.broadcast_to_rank_sync(rank_id, "_nb_hash", 0, &_nb_hash, sizeof(_nb_hash));
			broadcast_mode_sync(rank_id, BloomMode::BLOOM_INIT);

			int nb_dpus_in_rank = _pim_rankset.get_nb_dpu_in_rank(rank_id);
			auto uids = std::vector<uint64_t>(nb_dpus_in_rank, 0);
			for (int i = 0; i < nb_dpus_in_rank; i++) {
				uids[i] = DPU_UID(rank_id, i);
			}
			_pim_rankset.send_data_to_rank_sync(rank_id, "_dpu_uid", 0, uids, sizeof(uint64_t));

			_pim_rankset.launch_rank_sync(rank_id);
		}, true);

	}

	~PimBloomFilter() = default;

	void insert(const std::vector<uint64_t>& items) {

		const size_t nb_items = items.size();
		const size_t nb_ranks = _pim_rankset.get_nb_ranks();
		const size_t nb_workers = 4;

		const uint64_t max_nb_items_per_bucket = MAX_NB_ITEMS_PER_DPU / nb_ranks;
		const size_t bucket_size = MAX_NB_ITEMS_PER_DPU + 1;//CEIL8(max_nb_items_per_bucket + 1);
		const size_t bucket_length = sizeof(uint64_t) * bucket_size;

		auto statistics = LaunchStatistics();

		auto done_data = std::vector<std::vector<std::vector<uint64_t>>>();
		// Maybe reserve
		std::mutex done_data_mutex;
		std::mutex debug; // TODO remove me

		#pragma omp parallel num_threads(nb_workers)
		{

			int worker_id = omp_get_thread_num();
			auto rank_buckets = std::vector<std::vector<std::vector<uint64_t>>>(nb_ranks);
			
			// Consider a partition of the items
			for (size_t i = worker_id; i < nb_items; i += nb_workers) {
				uint64_t item = items[i];
				auto dispatch_data = _get_rank_dpu_id_from_item(item);
				uint32_t rank_id = dispatch_data.first;
				uint32_t bucket_idx = dispatch_data.second;
				size_t nb_dpus_in_rank = _pim_rankset.get_nb_dpu_in_rank(rank_id);

				auto &buckets = rank_buckets[rank_id];

				if (buckets.empty()) {
					buckets.resize(nb_dpus_in_rank);
					for (size_t d = 0; d < nb_dpus_in_rank; d++) {
						buckets[d].resize(bucket_size, 0);
					}
				}
				
				std::vector<uint64_t> &bucket = buckets[bucket_idx];
				bucket[0]++;
				bucket[bucket[0]] = item;

				debug.lock();
				if (bucket[0] >= max_nb_items_per_bucket) {
					done_data_mutex.lock();
					done_data.push_back(std::move(buckets));
					_insert_launch(rank_id, done_data.back(), bucket_length, statistics);
					done_data_mutex.unlock();
					rank_buckets[rank_id] = std::vector<std::vector<uint64_t>>();
				}
				debug.unlock();
				

			}
			
			
			// Launch remaining buckets that are not full
			debug.lock();
			for (size_t rank_id = 0; rank_id < nb_ranks; rank_id++) {
				auto &buckets = rank_buckets[rank_id];
				if (!buckets.empty()) {
					done_data_mutex.lock();
					done_data.push_back(std::move(buckets));
					_insert_launch(rank_id, done_data.back(), bucket_length, statistics);
					done_data_mutex.unlock();
				}
			}
			debug.unlock();

			// Call to see on trace when workers are done
			_worker_done();

			
		}
		
		_pim_rankset.wait_all_ranks_done();
		statistics.print();

	}

	uint32_t contains(const std::vector<uint64_t>& items) {

		(void) items; // Not implemented yet

		uint32_t total_nb_positive_lookups = 0;
		// int nb_dpu = _pim_rankset.get_nb_dpu();

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

		// 		int start_idx = _pim_rankset.get_cum_dpu_idx_for_rank(rank_id);
		// 		int nb_dpu_in_rank = _pim_rankset.get_nb_dpu_in_rank(rank_id);

		// 		// Launch
		// 		_pim_rankset.wait_rank_ready(rank_id);
		// 		_pim_rankset.send_data_to_rank<uint64_t>(rank_id, "items", 0,
		// 			std::vector<uint64_t*>(buckets.begin() + start_idx, buckets.begin() + start_idx + nb_dpu_in_rank),
		// 			sizeof(uint64_t) * CEIL8(MAX_NB_ITEMS_PER_DPU + 1));
		// 		broadcast_mode(rank_id, BloomMode::BLOOM_LOOKUP);
		// 		_pim_rankset.launch_rank_sync(rank_id);

		// 		total_nb_positive_lookups += _pim_rankset.get_reduced_sum_from_rank<uint32_t>(rank_id, "result", 0, sizeof(uint32_t));

		// 		// Reset buckets
		// 		for (int d = start_idx; d < (start_idx + nb_dpu_in_rank); d++) {
		// 			buckets[d][0] = 0;
		// 		}

		// 	}
		// 	buckets[bucket_idx][0]++;
 		// 	buckets[bucket_idx][buckets[bucket_idx][0]] = item;
		// }

		// // Launch a last round for the ranks that need it
		// for (int rank_id = 0; rank_id < _pim_rankset.get_nb_ranks(); rank_id++) {
		// 	int start_idx = _pim_rankset.get_cum_dpu_idx_for_rank(rank_id);
		// 	int nb_dpu_in_rank = _pim_rankset.get_nb_dpu_in_rank(rank_id);
		// 	for (int d = start_idx; d < (start_idx + nb_dpu_in_rank); d++) {
		// 		if (buckets[d][0]) {
		// 			_pim_rankset.wait_rank_ready(rank_id);
		// 			_pim_rankset.send_data_to_rank<uint64_t>(rank_id, "items", 0,
		// 				std::vector<uint64_t*>(buckets.begin() + start_idx, buckets.begin() + start_idx + nb_dpu_in_rank),
		// 				sizeof(uint64_t) * CEIL8(MAX_NB_ITEMS_PER_DPU + 1));
		// 			broadcast_mode(rank_id, BloomMode::BLOOM_LOOKUP);
		// 			_pim_rankset.launch_rank_sync(rank_id);
		// 			total_nb_positive_lookups += _pim_rankset.get_reduced_sum_from_rank<uint32_t>(rank_id, "result", 0, sizeof(uint32_t));
		// 		}
		// 	}
		// }
		return total_nb_positive_lookups;
	}

	/// @brief Computes the weight of the filter
	/// @return number of bits set to 1 in the filter
	uint32_t get_weight() {

		// Launch in parallel all ranks
		_pim_rankset.for_each_rank([this](size_t rank_id) {
			broadcast_mode_async(rank_id, BloomMode::BLOOM_WEIGHT);
			_pim_rankset.launch_rank_async(rank_id);
		}, true);

		// Reduce results in sequential
		uint32_t weight = 0;
		_pim_rankset.for_each_rank([this, &weight](size_t rank_id) {
			_pim_rankset.wait_rank_done(rank_id);
			weight += _pim_rankset.get_reduced_sum_from_rank_sync<uint32_t>(rank_id, "result", 0, sizeof(uint32_t));
		}, false);

		return weight;

	}

	// Wrappers for single items

	/// @brief Inserts a single item in the filter
	/// @param item item to insert
	void insert (const uint64_t& item) { auto items = std::vector<uint64_t>{item}; insert(items); }

	uint32_t contains (const uint64_t& item)  { return contains(std::vector<uint64_t>{item}); }

private:

	PimRankSet _pim_rankset;
	int _size2;
	uint64_t _size;
	uint64_t _size_reduced;
	int _dpu_size2;
	int _nb_hash;
	BloomHashFunctors _hash_functions;

	class LaunchStatistics {

		public:

			void incr_rounds(size_t more_items) {
				_update_mutex.lock();
				_items_handled += more_items;
				_rounds_lauched++;
				_update_mutex.unlock();
			}

			void print() {
				spdlog::info("Launched {} rounds and handled {} items", _rounds_lauched, _items_handled);
			}

		private:

			std::mutex _update_mutex;

			size_t _rounds_lauched = 0;
			size_t _items_handled = 0;
		
	};

	const std::string get_dpu_binary_name() {
		return std::string(DPU_BINARIES_DIR) + "/bloom_filters_dpu";
	}

	void broadcast_mode_sync(size_t rank_id, BloomMode mode) {
		_pim_rankset.broadcast_to_rank_sync(rank_id, "_mode", 0, &mode, sizeof(mode));
	}

	void broadcast_mode_async(size_t rank_id, BloomMode mode) {
		_pim_rankset.broadcast_to_rank_async(rank_id, "_mode", 0, &mode, sizeof(mode));
	}

	static inline uint32_t fastrange32(uint32_t word, uint32_t p) {
		return (uint32_t)(((uint64_t)word * (uint64_t)p) >> 32);
	}

	std::pair<uint32_t, uint32_t> _get_rank_dpu_id_from_item(uint64_t item) {
		uint64_t h0 = this->_hash_functions(item, 0);
		uint32_t rank_id = fastrange32(h0, _pim_rankset.get_nb_ranks());
		size_t nb_dpus_in_rank = _pim_rankset.get_nb_dpu_in_rank(rank_id);
		uint64_t h1 = this->_hash_functions(item, 1); // IMPORTANT: use a different hash otherwise some correlation can happen and some indexes may never be picked
		uint32_t dpu_id = fastrange32(h1, nb_dpus_in_rank);
		return std::pair<uint32_t, uint32_t>(rank_id, dpu_id);
	}

	void _insert_launch(size_t rank_id, std::vector<std::vector<uint64_t>>& buckets, const size_t &bucket_length, LaunchStatistics& statistics) {

		uint64_t items_sent = 0;
		for (auto bucket : buckets) {
			items_sent += bucket[0];
		}

		_pim_rankset.lock_rank(rank_id); // Lock so that other workers don't stack async calls in-between
		
		_pim_rankset.send_data_to_rank_async<uint64_t>(rank_id, "items", 0, buckets, bucket_length);
		broadcast_mode_async(rank_id, BloomMode::BLOOM_INSERT);

		// Error check
		auto check_pair = new std::pair<PimRankSet*, uint64_t>(&_pim_rankset, items_sent);
		auto check_callback_data = new PimCallbackData(rank_id, check_pair, [](size_t rank_id, void* arg) {
			auto check_pair = static_cast<std::pair<PimRankSet*, uint64_t>*>(arg);
			uint64_t items_received = check_pair->first->get_reduced_sum_from_rank_sync<uint64_t>(rank_id, "items", 0, sizeof(uint64_t));
			uint64_t items_sent = check_pair->second;
			if (items_sent != items_received) {
				spdlog::error("Rank {}: DPUs received {} items instead of {} (diff = {})", rank_id, items_received, items_sent, (items_received - items_sent));
			} else {
				spdlog::debug("Rank {}: DPUs received the right amount of items ({})", rank_id, items_received);
			}
			delete check_pair;
		});
		_pim_rankset.add_callback_async(check_callback_data);
		//

		// Launch
		_pim_rankset.launch_rank_async(rank_id);

		_pim_rankset.unlock_rank(rank_id);
		spdlog::info("Planned to launch rank {}", rank_id);
		statistics.incr_rounds(items_sent);

	}

};