#include <cmath>
#include <iostream>
#include <sstream>
#include <mutex>
#include <queue>
#include <thread>

#include "pim_rankset.cpp"
#include "pim_bloom_filter_common.h"

#include"bloom_filter.hpp"

void _worker_done() {}

/* -------------------------------------------------------------------------- */
/*                              Item dispatchers                              */
/* -------------------------------------------------------------------------- */

class HashPimItemDispatcher : public PimDispatcher<uint64_t> {

	public:

		HashPimItemDispatcher(PimRankSet& pim_rankset) : PimDispatcher<uint64_t>(pim_rankset), _hash_functions(2) {}

		PimUnitUID dispatch(const uint64_t& item) override {
			uint64_t h0 = _hash_functions(item, 0);
			size_t rank_id = fastrange32(h0, get_pim_rankset().get_nb_ranks());
			size_t nb_dpus_in_rank = get_pim_rankset().get_nb_dpu_in_rank(rank_id);
			uint64_t h1 = _hash_functions(item, 1); // IMPORTANT: use a different hash otherwise some correlation can happen and some indexes may never be picked
			size_t dpu_id = fastrange32(h1, nb_dpus_in_rank);
			return PimUnitUID(rank_id, dpu_id);
		}
	
	private:

		BloomHashFunctions _hash_functions;

		static inline size_t fastrange32(uint32_t key, uint32_t idx) {
			return static_cast<size_t>((static_cast<uint64_t>(key) * static_cast<uint64_t>(idx)) >> 32);
		}


};


/* -------------------------------------------------------------------------- */
/*                              PIM Bloom filter                              */
/* -------------------------------------------------------------------------- */

template <class ItemDispatcher>
class PimBloomFilter : public BulkBloomFilter {

	public:

		PimBloomFilter(	size_t size2,
						size_t nb_hash,
						size_t nb_threads,
						size_t nb_ranks = 8,
						DpuProfile dpu_profile = DpuProfile::HARDWARE) : BulkBloomFilter(size2, nb_hash, nb_threads),
							_pim_rankset(PimRankSet(nb_ranks, nb_threads, dpu_profile, get_dpu_binary_name().c_str())),
							_item_dispatcher(HashPimItemDispatcher(_pim_rankset)) {

			static_assert(std::is_base_of<PimDispatcher<uint64_t>, ItemDispatcher>::value, "type parameter of this class must derive from PimDispatcher");

			if (nb_ranks < 1) {
				throw std::invalid_argument(std::string("Error: number of DPUs ranks must be >= 1"));
			}

			if (_get_size() < _pim_rankset.get_nb_dpu()) {
				throw std::invalid_argument(std::string("Error: Asking too little space per DPU, use less DPUs or a bigger filter"));
			}

			_dpu_size2 = ceil(log(_get_size() / (_pim_rankset.get_nb_dpu() * 16)) / log(2));

			if ((_dpu_size2 >> 3) > MAX_BLOOM_DPU_SIZE2) {
				throw std::invalid_argument(
					std::string("Error: filter size2 per DPU is bigger than max space available (")
					+ std::to_string(_dpu_size2)
					+ std::string(" > ")
					+ std::to_string(MAX_BLOOM_DPU_SIZE2)
					+ std::string("), try to reduce the size or increase the number of DPUs")
				);
			}

			size_t nb_hash_functions = get_nb_hash();
			_pim_rankset.for_each_rank([this, nb_hash_functions](size_t rank_id) {
				_pim_rankset.broadcast_to_rank_sync(rank_id, "dpu_size2", 0, &_dpu_size2, sizeof(_dpu_size2));
				_pim_rankset.broadcast_to_rank_sync(rank_id, "nb_hash", 0, &nb_hash_functions, sizeof(nb_hash_functions));
				broadcast_mode_sync(rank_id, INIT_TKN);

				int nb_dpus_in_rank = _pim_rankset.get_nb_dpu_in_rank(rank_id);
				auto uids = std::vector<size_t>(nb_dpus_in_rank, 0);
				for (int i = 0; i < nb_dpus_in_rank; i++) {
					uids[i] = DPU_UID(rank_id, i);
				}
				_pim_rankset.send_data_to_rank_sync(rank_id, "dpu_uid", 0, uids, sizeof(size_t));

				_pim_rankset.launch_rank_sync(rank_id);
			}, true);

		}

		void insert_bulk(const std::vector<uint64_t>& items) override {

			const size_t nb_items = items.size();
			const size_t nb_ranks = _pim_rankset.get_nb_ranks();
			const size_t nb_workers = 8;

			const uint64_t max_nb_items_per_bucket = MAX_NB_ITEMS_PER_DPU / nb_ranks;
			const size_t bucket_size = max_nb_items_per_bucket + 1;
			const size_t bucket_length = sizeof(uint64_t) * bucket_size;

			auto statistics = LaunchStatistics();

			auto done_containers = std::vector<std::vector<std::vector<std::vector<uint64_t>>>>(nb_workers);
			for (auto done_container : done_containers) {
				done_container.reserve(nb_items / max_nb_items_per_bucket / nb_workers);
			}

			#pragma omp parallel num_threads(nb_workers)
			{

				int worker_id = omp_get_thread_num();
				auto rank_buckets = std::vector<std::vector<std::vector<uint64_t>>>(nb_ranks);
				
				// Consider a partition of the items
				for (size_t i = worker_id; i < nb_items; i += nb_workers) {
					uint64_t item = items[i];
					auto dispatch_data = _item_dispatcher.dispatch(item);
					size_t rank_id = dispatch_data.get_rank_id();
					size_t bucket_idx = dispatch_data.get_dpu_id();
					size_t nb_dpus_in_rank = _pim_rankset.get_nb_dpu_in_rank(rank_id);

					auto &buckets = rank_buckets[rank_id];
					
					if (buckets.empty()) {
						buckets.resize(nb_dpus_in_rank);
						for (auto &bucket : buckets) {
							bucket.resize(bucket_size, 0);
						}
					}
					
					std::vector<uint64_t> &bucket = buckets[bucket_idx];
					bucket[0]++;
					bucket[bucket[0]] = item;

					if ((bucket[0] >= max_nb_items_per_bucket)) {
						done_containers[worker_id].push_back(std::move(buckets));
						_insert_launch(rank_id, done_containers[worker_id].back(), bucket_length, statistics);
						rank_buckets[rank_id] = std::vector<std::vector<uint64_t>>();
					}
					
				}
				
				// Launch remaining buckets that are not full
				for (size_t rank_id = 0; rank_id < nb_ranks; rank_id++) {
					auto &buckets = rank_buckets[rank_id];
					if (!buckets.empty()) {
						done_containers[worker_id].push_back(std::move(buckets));
						_insert_launch(rank_id, done_containers[worker_id].back(), bucket_length, statistics);
					}
				}

				// Call to see on trace when workers are done
				_worker_done();
				
			}
			
			_pim_rankset.wait_all_ranks_done();

			if (spdlog::default_logger_raw()->level() == spdlog::level::debug) {
				statistics.print();
			}

		}

		std::vector<bool> contains_bulk(const std::vector<uint64_t>& items) override {
			(void) items; // Not implemented yet
			return std::vector<bool>(items.size(), false); // FIXME
		}

		size_t get_weight() override {

			// Launch in parallel all ranks
			_pim_rankset.for_each_rank([this](size_t rank_id) {
				broadcast_mode_async(rank_id, WEIGHT_TKN);
				_pim_rankset.launch_rank_async(rank_id);
			}, true);

			// Reduce results in sequential
			uint64_t weight = 0;
			_pim_rankset.for_each_rank([this, &weight](size_t rank_id) {
				_pim_rankset.wait_rank_done(rank_id);
				weight += _pim_rankset.get_reduced_sum_from_rank_sync<uint64_t>(rank_id, "result", 0, sizeof(uint64_t));
			}, false);

			return static_cast<size_t>(weight);
		
		}

		const std::vector<uint8_t>& get_data() override {
			return _bloom_data;
			// FIXME: not implemented yet
		}

	private:

		PimRankSet _pim_rankset;
		size_t _dpu_size2;
		HashPimItemDispatcher _item_dispatcher;
		std::vector<uint8_t> _bloom_data;

		class LaunchStatistics {

			public:

				void incr_rounds(size_t more_items) {
					_update_mutex.lock();
					_items_handled += more_items;
					_rounds_lauched++;
					_update_mutex.unlock();
				}

				void print() {
					spdlog::debug("Launched {} rounds and handled {} items", _rounds_lauched, _items_handled);
				}

			private:

				std::mutex _update_mutex;

				size_t _rounds_lauched = 0;
				size_t _items_handled = 0;
			
		};

		const std::string get_dpu_binary_name() {
			return std::string(DPU_BINARIES_DIR) + "/bloom_filters_dpu";
		}

		// To have mode values stored somewhere for async transfer
		BloomMode INIT_TKN = BloomMode::BLOOM_INIT;
		BloomMode INSERT_TKN = BloomMode::BLOOM_INSERT;
		BloomMode LOOKUP_TKN = BloomMode::BLOOM_LOOKUP;
		BloomMode WEIGHT_TKN = BloomMode::BLOOM_WEIGHT;

		void broadcast_mode_sync(size_t rank_id, BloomMode& mode) {
			_pim_rankset.broadcast_to_rank_sync(rank_id, "mode", 0, &mode, sizeof(mode));
		}

		void broadcast_mode_async(size_t rank_id, BloomMode& mode) {
			_pim_rankset.broadcast_to_rank_async(rank_id, "mode", 0, &mode, sizeof(mode));
		}

		void _insert_launch(size_t rank_id, std::vector<std::vector<uint64_t>>& buckets, const size_t &bucket_length, LaunchStatistics& statistics) {

			_pim_rankset.lock_rank(rank_id); // Lock so that other workers don't stack async calls in-between

			_pim_rankset.send_data_to_rank_async<uint64_t>(rank_id, "items", 0, buckets, bucket_length);
			broadcast_mode_async(rank_id, INSERT_TKN);

			// Error checking for number of items
			// uint64_t items_sent = 0;
			// for (auto &bucket : buckets) {
			// 	items_sent += bucket[0];
			// }

			// _pim_rankset.add_callback_async(rank_id, &_pim_rankset, [items_sent](size_t rank_id, void* arg) {
			// 	auto rankset = static_cast<PimRankSet*>(arg);
			// 	uint64_t items_received = rankset->get_reduced_sum_from_rank_sync<uint64_t>(rank_id, "items", 0, sizeof(uint64_t));
			// 	if (items_sent != items_received) {
			// 		spdlog::error("Rank {}: DPUs received {} items instead of {} (diff = {})", rank_id, items_received, items_sent, (items_received - items_sent));
			// 	} else {
			// 		spdlog::debug("Rank {}: DPUs received the right amount of items ({})", rank_id, items_received);
			// 	}
			// });
			// ----------------- //

			_pim_rankset.launch_rank_async(rank_id);

			_pim_rankset.unlock_rank(rank_id);

			spdlog::debug("Stacked calls to launch rank {}", rank_id);

			if (spdlog::default_logger_raw()->level() == spdlog::level::debug) {
				uint64_t items_sent = 0;
				for (auto &bucket : buckets) {
					items_sent += bucket[0];
				}
				statistics.incr_rounds(items_sent);
			}

		}

};