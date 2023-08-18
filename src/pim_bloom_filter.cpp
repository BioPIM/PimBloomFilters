#include <cmath>
#include <iostream>
#include <sstream>
#include <mutex>
#include <queue>
#include <thread>

#include "pim_rankset.cpp"
#include "pim_bloom_filter_common.h"

#include"bloom_filter.hpp"

void __attribute__((optimize(0))) _worker_done() {}

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

			_dpu_size2 = ceil(log(_get_size() / (_pim_rankset.get_nb_dpu() * NR_TASKLETS)) / log(2));

			if ((_dpu_size2 >> 3) > MAX_BLOOM_DPU_SIZE2) {
				throw std::invalid_argument(
					std::string("Error: filter size2 per DPU is bigger than max space available (")
					+ std::to_string(_dpu_size2)
					+ std::string(" > ")
					+ std::to_string(MAX_BLOOM_DPU_SIZE2)
					+ std::string("), try to reduce the size or increase the number of DPUs")
				);
			}

			auto args = std::vector<uint64_t>{BloomFunction::BLOOM_INIT, _dpu_size2, get_nb_hash()};
			_pim_rankset.for_each_rank([this, args](size_t rank_id) {
				_pim_rankset.broadcast_to_rank_sync(rank_id, "args", 0, args);

				// int nb_dpus_in_rank = _pim_rankset.get_nb_dpu_in_rank(rank_id);
				// auto uids = std::vector<size_t>(nb_dpus_in_rank, 0);
				// for (int i = 0; i < nb_dpus_in_rank; i++) {
				// 	uids[i] = DPU_UID(rank_id, i);
				// }
				// _pim_rankset.send_data_to_rank_sync(rank_id, "dpu_uid", 0, uids, sizeof(size_t));

				_pim_rankset.launch_rank_sync(rank_id);
			}, true);

		}

		void insert_bulk(const std::vector<uint64_t>& items) override {

			const size_t nb_items = items.size();
			const size_t nb_ranks = _pim_rankset.get_nb_ranks();
			const size_t nb_workers = 8;

			const uint64_t max_nb_items_per_bucket = MAX_NB_ITEMS_PER_DPU / nb_ranks;
			const size_t bucket_size = max_nb_items_per_bucket + 2;
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
				auto &done_container = done_containers[worker_id];
				
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
							bucket[0] = BloomFunction::BLOOM_INSERT;
						}
					}
					
					auto &bucket = buckets[bucket_idx];
					bucket[1]++;
					bucket[bucket[1] + 1] = item;

					if ((bucket[1] >= max_nb_items_per_bucket)) {
						done_container.push_back(std::move(buckets));
						_insert_launch(rank_id, done_container.back(), bucket_length, statistics);
						rank_buckets[rank_id] = std::vector<std::vector<uint64_t>>();
					}
					
				}
				
				// Launch remaining buckets that are not full
				for (size_t rank_id = 0; rank_id < nb_ranks; rank_id++) {
					auto &buckets = rank_buckets[rank_id];
					if (!buckets.empty()) {
						done_container.push_back(std::move(buckets));
						_insert_launch(rank_id, done_container.back(), bucket_length, statistics);
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

			const size_t nb_items = items.size();
			const size_t nb_ranks = _pim_rankset.get_nb_ranks();
			const size_t nb_workers = 8;

			const uint64_t max_nb_items_per_bucket = MAX_NB_ITEMS_PER_DPU / nb_ranks;
			const size_t bucket_size = max_nb_items_per_bucket + 2;
			const size_t bucket_length = sizeof(uint64_t) * bucket_size;

			auto statistics = LaunchStatistics();

			auto done_containers = std::vector<std::vector<std::vector<std::vector<uint64_t>>>>(nb_workers);
			auto indexes_containers = std::vector<std::vector<std::vector<std::vector<size_t>>>>(nb_workers);
			for (auto done_container : done_containers) {
				size_t estimated_size = nb_items / max_nb_items_per_bucket / nb_workers;
				done_container.reserve(estimated_size);
				indexes_containers.reserve(estimated_size);
			}

			// Editing a vector of bool is not thread-safe even if accessing different cells, so using an intermediate vector
			auto int_result = std::vector<int>(items.size(), false);

			#pragma omp parallel num_threads(nb_workers)
			{

				int worker_id = omp_get_thread_num();
				auto rank_buckets = std::vector<std::vector<std::vector<uint64_t>>>(nb_ranks);
				auto rank_indexes_buckets = std::vector<std::vector<std::vector<size_t>>>(nb_ranks);
				auto &done_container = done_containers[worker_id];
				auto &indexes_container = indexes_containers[worker_id];
				
				// Consider a partition of the items
				for (size_t i = worker_id; i < nb_items; i += nb_workers) {
					uint64_t item = items[i];
					auto dispatch_data = _item_dispatcher.dispatch(item);
					size_t rank_id = dispatch_data.get_rank_id();
					size_t bucket_idx = dispatch_data.get_dpu_id();
					size_t nb_dpus_in_rank = _pim_rankset.get_nb_dpu_in_rank(rank_id);

					auto &buckets = rank_buckets[rank_id];
					auto &indexes_buckets = rank_indexes_buckets[rank_id];
					
					if (buckets.empty()) {
						buckets.resize(nb_dpus_in_rank);
						indexes_buckets.resize(nb_dpus_in_rank);
						for (auto &bucket : buckets) {
							bucket.resize(bucket_size, 0);
							bucket[0] = BloomFunction::BLOOM_LOOKUP;
						}
						for (auto &bucket : indexes_buckets) {
							bucket.resize(bucket_size - 1, 0);
						}
					}
					
					auto &bucket = buckets[bucket_idx];
					auto &indexes_bucket = indexes_buckets[bucket_idx];
					bucket[1]++;
					indexes_bucket[0]++;
					bucket[bucket[1] + 1] = item;
					indexes_bucket[bucket[1]] = i;

					if ((bucket[1] >= max_nb_items_per_bucket)) {
						done_container.push_back(std::move(buckets));
						indexes_container.push_back(std::move(indexes_buckets));
						_contains_launch(rank_id, done_container.back(), bucket_length, int_result, indexes_container.back(), statistics);
						rank_buckets[rank_id] = std::vector<std::vector<uint64_t>>();
						rank_indexes_buckets[rank_id] = std::vector<std::vector<size_t>>();
					}
					
				}
				
				// Launch remaining buckets that are not full
				for (size_t rank_id = 0; rank_id < nb_ranks; rank_id++) {
					auto &buckets = rank_buckets[rank_id];
					auto &indexes_buckets = rank_indexes_buckets[rank_id];
					if (!buckets.empty()) {
						done_container.push_back(std::move(buckets));
						indexes_container.push_back(std::move(indexes_buckets));
						_contains_launch(rank_id, done_container.back(), bucket_length, int_result, indexes_container.back(), statistics);
					}
				}

				// Call to see on trace when workers are done
				_worker_done();
				
			}
			
			_pim_rankset.wait_all_ranks_done();

			if (spdlog::default_logger_raw()->level() == spdlog::level::debug) {
				statistics.print();
			}

			// Formatting back into a vector of bool
            auto result =  std::vector<bool>(int_result.size(), false);
            for (size_t i = 0; i < int_result.size(); i++) {
                result[i] = int_result[i];
            }

            return result;
		}

		size_t get_weight() override {

			uint64_t weight = 0;
			std::mutex weight_mutex;

			auto callback = [&weight, &weight_mutex](size_t rank_id, void* arg) {
				auto rankset = static_cast<PimRankSet*>(arg);
				auto result = rankset->get_reduced_sum_from_rank_sync<uint64_t>(rank_id, "result", 0, sizeof(uint64_t));
				weight_mutex.lock();
				weight += result;
				weight_mutex.unlock();
			};

			auto args = std::vector<uint64_t>{BloomFunction::BLOOM_WEIGHT};

			_pim_rankset.for_each_rank([this, &callback, &args](size_t rank_id) {
				_pim_rankset.broadcast_to_rank_async(rank_id, "args", 0, args);
				_pim_rankset.launch_rank_async(rank_id);
				_pim_rankset.add_callback_async(rank_id, &_pim_rankset, callback);
			}, true);

			_pim_rankset.wait_all_ranks_done();
			return static_cast<size_t>(weight);

		}

		const std::vector<uint8_t>& get_data() override {
			// NB: for ease we get the whole buffer no matter the actual filter size
			// because it would be a bit trickier to get only the relevant part used by each tasklet
			// So the data returned takes (a lot) more storage than it could
			// But works fine this way for now
			_bloom_data.resize(0);
			_bloom_data.reserve(MAX_BLOOM_DPU_SIZE * NR_TASKLETS * _pim_rankset.get_nb_dpu());
			_pim_rankset.for_each_rank([this](size_t rank_id) {
				auto rank_data = _pim_rankset.get_vec_data_from_rank_sync<uint8_t>(rank_id, "_bloom_data", 0, MAX_BLOOM_DPU_SIZE * NR_TASKLETS * sizeof(uint8_t));
				for (auto &dpu_data : rank_data) {
					std::move(dpu_data.begin(), dpu_data.end(), std::back_inserter(_bloom_data));
				}
			}, false); // Sequential because we need the order to be deterministic
			return _bloom_data;
		}

		void set_data(const std::vector<uint8_t>& data) override {
			size_t start_index = 0;
			auto rank_buffers = std::vector<std::vector<std::vector<uint8_t>>>(_pim_rankset.get_nb_ranks());
			_pim_rankset.for_each_rank([this, &start_index, &data, &rank_buffers](size_t rank_id) {
				size_t nb_dpu_in_rank = _pim_rankset.get_nb_dpu_in_rank(rank_id);
				auto &buffers = rank_buffers[rank_id];
				buffers.resize(nb_dpu_in_rank);
				for (size_t dpu_id = 0; dpu_id < nb_dpu_in_rank; dpu_id++) {
					buffers[dpu_id].assign(data.begin() + start_index, data.begin() + start_index + MAX_BLOOM_DPU_SIZE * NR_TASKLETS);
					start_index += MAX_BLOOM_DPU_SIZE * NR_TASKLETS;
				}
				_pim_rankset.send_data_to_rank_async<uint8_t>(rank_id, "_bloom_data", 0, buffers , MAX_BLOOM_DPU_SIZE * NR_TASKLETS * sizeof(uint8_t));
			}, false); // Sequential because we need to restore in the same order it was got (but calls can be async)
			_pim_rankset.wait_all_ranks_done();
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

		void _insert_launch(size_t rank_id, std::vector<std::vector<uint64_t>>& buckets, const size_t& bucket_length, LaunchStatistics& statistics) {

			_pim_rankset.lock_rank(rank_id); // Lock so that other workers don't stack async calls in-between

			_pim_rankset.send_data_to_rank_async<uint64_t>(rank_id, "args", 0, buckets, bucket_length);

			// Error checking for number of items
			// uint64_t items_sent = 0;
			// for (auto &bucket : buckets) {
			// 	items_sent += bucket[0];
			// }

			// _pim_rankset.add_callback_async(rank_id, &_pim_rankset, [items_sent](size_t rank_id, void* arg) {
			// 	auto rankset = static_cast<PimRankSet*>(arg);
			// 	auto items_received = rankset->get_reduced_sum_from_rank_sync<uint64_t>(rank_id, "items", 0, sizeof(uint64_t));
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

			(void) statistics;
			// if (spdlog::default_logger_raw()->level() == spdlog::level::debug) {
			// 	uint64_t items_sent = 0;
			// 	for (auto &bucket : buckets) {
			// 		items_sent += bucket[1];
			// 	}
			// 	statistics.incr_rounds(items_sent);
			// }

		}

		void _contains_launch(size_t rank_id, std::vector<std::vector<uint64_t>>& buckets, const size_t& bucket_length, std::vector<int>& lookup_results, std::vector<std::vector<size_t>>& indexes_buckets, LaunchStatistics& statistics) {

			_pim_rankset.lock_rank(rank_id); // Lock so that other workers don't stack async calls in-between

			_pim_rankset.send_data_to_rank_async<uint64_t>(rank_id, "args", 0, buckets, bucket_length);

			_pim_rankset.launch_rank_async(rank_id);
			
			// Get results
			// NB: indexes_buckets doesn't work with reference capture for some reason
			_pim_rankset.add_callback_async(rank_id, &_pim_rankset, [indexes_buckets, bucket_length, &lookup_results](size_t rank_id, void* arg) {
				auto rankset = static_cast<PimRankSet*>(arg);
				auto rank_results = rankset->get_vec_data_from_rank_sync<uint64_t>(rank_id, "args", 0, bucket_length);
				for (size_t dpu_id = 0; dpu_id < indexes_buckets.size(); dpu_id++) {
					auto &indexes_bucket = indexes_buckets[dpu_id];
					auto &bucket_results = rank_results[dpu_id];
					for (size_t i = 1; i <= indexes_bucket[0]; i++) {
						lookup_results[indexes_bucket[i]] = bucket_results[i + 1];
					}
				}
			});

			_pim_rankset.unlock_rank(rank_id);

			spdlog::debug("Stacked calls to launch rank {}", rank_id);

			(void) statistics;
			// if (spdlog::default_logger_raw()->level() == spdlog::level::debug) {
			// 	uint64_t items_sent = 0;
			// 	for (auto &bucket : buckets) {
			// 		items_sent += bucket[1];
			// 	}
			// 	statistics.incr_rounds(items_sent);
			// }

		}

};