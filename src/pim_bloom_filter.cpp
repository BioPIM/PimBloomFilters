#include <cmath>
#include <iostream>
#include <sstream>
#include <mutex>
#include <queue>
#include <thread>

#include "pim_bloom_filter_common.h"
#include "pim_rankset.cpp"

#include"bloom_filter.hpp"

void __attribute__((optimize(0))) __worker_done() {}
void __attribute__((optimize(0))) __method_start() {}
void __attribute__((optimize(0))) __method_end() {}


/* -------------------------------------------------------------------------- */
/*                              Item dispatchers                              */
/* -------------------------------------------------------------------------- */

class HashPimItemDispatcher : public PimDispatcher<uint64_t> {

	public:

		HashPimItemDispatcher(PimRankSet& pim_rankset) : PimDispatcher<uint64_t>(pim_rankset), _hash_functions(2) {}

		PimUnitUID dispatch(const uint64_t& item) override {
			uint64_t h0 = _hash_functions(item, 0);
			// Use higher 32 bits for rank and lower 32 bits for dpu inside a rank
			size_t rank_id = fastrange32(static_cast<uint32_t>(h0 >> 32), get_pim_rankset().get_nb_ranks());
			size_t dpu_id = fastrange32(static_cast<uint32_t>(h0), get_pim_rankset().get_nb_dpu_in_rank(rank_id));
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
						DpuProfile dpu_profile = DpuProfile()) : BulkBloomFilter(size2, nb_hash, nb_threads),
							_pim_rankset(PimRankSet(nb_ranks, nb_threads, dpu_profile, std::string(DPU_BINARIES_DIR) + "/bloom_filters_dpu")),
							_item_dispatcher(HashPimItemDispatcher(_pim_rankset)) {

			static_assert(std::is_base_of<PimDispatcher<uint64_t>, ItemDispatcher>::value, "type parameter of this class must derive from PimDispatcher");

			if constexpr(DO_TRACE) { __method_start(); }

			if (nb_ranks < 1) {
				throw std::invalid_argument(std::string("Error: number of DPUs ranks must be >= 1"));
			}

			if (_get_size() < _pim_rankset.get_nb_dpu()) {
				throw std::invalid_argument(std::string("Error: Asking too little space per DPU, use less DPUs or a bigger filter"));
			}

			_dpu_size2 = ceil(log(_get_size() / (_pim_rankset.get_nb_dpu() * NR_TASKLETS)) / log(2));
			// spdlog::info("dpu_size2 is {}", _dpu_size2);

			if (_dpu_size2  > (MAX_BLOOM_DPU_SIZE2 + 3)) {
				throw std::invalid_argument(
					std::string("Error: filter size2 per DPU is bigger than max space available (")
					+ std::to_string(_dpu_size2)
					+ std::string(" > ")
					+ std::to_string(MAX_BLOOM_DPU_SIZE2 + 3)
					+ std::string("), try to reduce the size or increase the number of DPUs")
				);
			}
			
			std::vector<uint64_t> args{BloomFunction::BLOOM_INIT, _dpu_size2, get_nb_hash()};
			_pim_rankset.for_each_rank([this, &args](size_t rank_id) {
				_pim_rankset.broadcast_to_rank_sync<uint64_t>(rank_id, "args", 0, args);
				_pim_rankset.launch_rank_sync(rank_id);
			}, true);

			if constexpr(DO_TRACE) { __method_end(); }

		}

		void insert_bulk(const std::vector<uint64_t>& items) override {

			if constexpr(DO_TRACE) { __method_start(); }

			const size_t nb_items = items.size();
			const size_t nb_ranks = _pim_rankset.get_nb_ranks();
			const size_t nb_workers = 6;

			std::vector<double> workers_measures;
			if constexpr(DO_WORKLOAD_PROFILING) {
				workers_measures.resize(nb_workers);
				_pim_rankset.start_workload_profiling();
			}
			
			const size_t max_nb_items_per_bucket = MAX_NB_ITEMS_PER_DPU;
			const size_t bucket_size = max_nb_items_per_bucket + 2;
			const size_t bucket_length = sizeof(uint64_t) * bucket_size;

			auto done_containers = std::vector<std::vector<std::vector<std::vector<uint64_t>>>>(nb_workers);

			// IMPORTANT: program will likely crash if estimation here is lower than the actual count
			// Buckets will not be full all the same time and rank gets launched at first bucket full, so consider the optimal repartition
			// and increase it by 100% because we need to be sure no reallocation ever happens otherwise it will invalidate all references given to callbacks
			size_t estimated_size = (1.0 + nb_items / (_pim_rankset.get_nb_dpu() * max_nb_items_per_bucket * nb_workers)) * nb_ranks * 2.0;

			for (auto &done_container : done_containers) {
				done_container.reserve(estimated_size);
			}

			size_t D = nb_items / nb_workers;
			size_t K = nb_items % nb_workers;

			#pragma omp parallel num_threads(nb_workers)
			{

				size_t worker_id = omp_get_thread_num();
				auto buckets_done_idx = std::vector<size_t>(nb_ranks, _NO_MAPPING);
				auto &done_container = done_containers[worker_id];
				
				// Consider a partition of the items
				size_t start_index = worker_id * D + (worker_id < K ? worker_id : K);
				size_t stop_index = start_index + D + (worker_id < K ? 1 : 0);
				for (size_t i = start_index; i < stop_index; i++) { // Faster iteration
				// for (size_t i = worker_id; i < nb_items; i += nb_workers) { // Slower iteration, probably more memory-bound
					const uint64_t &item = items[i];
					auto dispatch_data = _item_dispatcher.dispatch(item);
					size_t rank_id = dispatch_data.get_rank_id();

					auto idx = buckets_done_idx[rank_id];

					if (idx != _NO_MAPPING) {
						auto &buckets = done_container[idx];
						auto &bucket = buckets[dispatch_data.get_dpu_id()];
						if (bucket.size() >= bucket_size) {
							// Set final size here instead of updating at each push_back to avoid writing to cells very far away each time
							for (auto &the_bucket : buckets) {
								the_bucket[1] = the_bucket.size() - 2;
							}
							_insert_launch(rank_id, buckets, bucket_length);
							idx = _NO_MAPPING; // These buckets are launched, forget the mapping so that other buckets get created later
						}
					}
					
					if (idx == _NO_MAPPING) {
						idx = done_container.size();
						buckets_done_idx[rank_id] = idx;
						auto &buckets = done_container.emplace_back();
						size_t nb_dpu_in_rank = _pim_rankset.get_nb_dpu_in_rank(rank_id);
						buckets.reserve(nb_dpu_in_rank);
						buckets.resize(nb_dpu_in_rank);
						for (auto &bucket : buckets) {
							bucket.reserve(bucket_size);
							bucket.emplace_back(BloomFunction::BLOOM_INSERT);
							bucket.emplace_back(0); // This cell will contain the number of items
						}
					}
					
					auto &buckets = done_container[idx];
					buckets[dispatch_data.get_dpu_id()].emplace_back(item);
					
				}
				
				// Launch remaining buckets that are not full
				for (size_t rank_id = 0; rank_id < nb_ranks; rank_id++) {
					auto idx = buckets_done_idx[rank_id];
					if (idx != _NO_MAPPING) {
						auto &buckets = done_container[idx];
						for (auto &the_bucket : buckets) {
							the_bucket[1] = the_bucket.size() - 2;
						}
						_insert_launch(rank_id, buckets, bucket_length);
					}
				}

				if constexpr(DO_WORKLOAD_PROFILING) { workers_measures[worker_id] = omp_get_wtime(); }
				if constexpr(DO_TRACE) { __worker_done(); }

				// spdlog::info("Worker {} did {} launches", worker_id, done_container.size());
				
			}
			
			_pim_rankset.wait_all_ranks_done();

			if constexpr(DO_WORKLOAD_PROFILING) {
				double stop = omp_get_wtime();
				for (size_t worker_id = 0; worker_id < nb_workers; worker_id++) {
					spdlog::info("Host worker {} had {} seconds of idle time", worker_id, stop - workers_measures[worker_id]);
				}
				_pim_rankset.end_workload_profiling();
			}

			if constexpr(DO_TRACE) { __method_end(); }

		}

		std::vector<bool> contains_bulk(const std::vector<uint64_t>& items) override {

			if constexpr(DO_TRACE) { __method_start(); }

			const size_t nb_items = items.size();
			const size_t nb_ranks = _pim_rankset.get_nb_ranks();
			const size_t nb_workers = 5;

			std::vector<double> workers_measures;
			if constexpr(DO_WORKLOAD_PROFILING) {
				workers_measures.resize(nb_workers);
				_pim_rankset.start_workload_profiling();
			}

			const uint64_t max_nb_items_per_bucket = MAX_NB_ITEMS_PER_DPU;
			const size_t bucket_size = max_nb_items_per_bucket + 2;
			const size_t bucket_length = sizeof(uint64_t) * bucket_size;

			auto done_containers = std::vector<std::vector<std::vector<std::vector<uint64_t>>>>(nb_workers);
			
			// IMPORTANT: program will likely crash if estimation here is lower than the actual count
			// Buckets will not be full all the same time and rank gets launched at first bucket full, so consider the optimal repartition
			// and increase it by 100% because we need to be sure no reallocation ever happens otherwise it will invalidate all references given to callbacks
			// Estimation is x2 compared to insert_bulk because we store twice more per launch (items + indexes)
			size_t estimated_size = (1.0 + nb_items / (_pim_rankset.get_nb_dpu() * max_nb_items_per_bucket * nb_workers)) * nb_ranks * 4.0;
			
			for (auto &done_container : done_containers) {
				done_container.reserve(estimated_size);
			}

			// Editing a vector of bool is not thread-safe even if accessing different cells, so using an intermediate vector
			auto int_result = std::vector<int>(items.size(), false);

			size_t D = nb_items / nb_workers;
			size_t K = nb_items % nb_workers;

			#pragma omp parallel num_threads(nb_workers)
			{

				size_t worker_id = omp_get_thread_num();
				auto buckets_done_idx = std::vector<size_t>(nb_ranks, _NO_MAPPING);
				auto &done_container = done_containers[worker_id];
				
				// Consider a partition of the items
				size_t start_index = worker_id * D + (worker_id < K ? worker_id : K);
				size_t stop_index = start_index + D + (worker_id < K ? 1 : 0);
				for (size_t i = start_index; i < stop_index; i++) { // Faster iteration
				// for (size_t i = worker_id; i < nb_items; i += nb_workers) {
					const uint64_t &item = items[i];
					auto dispatch_data = _item_dispatcher.dispatch(item);
					size_t rank_id = dispatch_data.get_rank_id();

					auto idx = buckets_done_idx[rank_id];

					if (idx != _NO_MAPPING) {
						auto &buckets = done_container[idx];
						auto &bucket = buckets[dispatch_data.get_dpu_id()];
						if (bucket.size() >= bucket_size) {
							for (auto &the_bucket : buckets) {
								the_bucket[1] = the_bucket.size() - 2;
							}
							auto &indexes_buckets = done_container[idx + 1];
							_contains_launch(rank_id, buckets, bucket_length, int_result, indexes_buckets);
							idx = _NO_MAPPING; // These buckets are launched, forget the mapping so that other buckets get created later
						}
					}
					
					if (idx == _NO_MAPPING) {
						idx = done_container.size();
						buckets_done_idx[rank_id] = idx;
						auto &buckets = done_container.emplace_back();
						auto &indexes_buckets = done_container.emplace_back();
						size_t nb_dpus_in_rank = _pim_rankset.get_nb_dpu_in_rank(rank_id);
						buckets.reserve(nb_dpus_in_rank);
						buckets.resize(nb_dpus_in_rank);
						indexes_buckets.reserve(nb_dpus_in_rank);
						indexes_buckets.resize(nb_dpus_in_rank);
						for (size_t i = 0; i < buckets.size(); i++) {
							buckets[i].reserve(bucket_size);
							buckets[i].emplace_back(BloomFunction::BLOOM_LOOKUP);
							buckets[i].emplace_back(0); // This cell will contain the number of items
							indexes_buckets[i].reserve(max_nb_items_per_bucket);
							// No cell to remember size in indexes_buckets, DPU will transfer the info to the host in its lookup results
						}
					}
					
					auto &buckets = done_container[idx];
					auto &indexes_buckets = done_container[idx + 1];
					size_t bucket_idx = dispatch_data.get_dpu_id();
					auto &bucket = buckets[bucket_idx];
					auto &indexes_bucket = indexes_buckets[bucket_idx];
					bucket.emplace_back(item);
					indexes_bucket.emplace_back(i);
					
				}
				
				// Launch remaining buckets that are not full
				for (size_t rank_id = 0; rank_id < nb_ranks; rank_id++) {
					auto idx = buckets_done_idx[rank_id];
					if (idx != _NO_MAPPING) {
						auto &buckets = done_container[idx];
						auto &indexes_buckets = done_container[idx + 1];
						for (auto &the_bucket : buckets) {
							the_bucket[1] = the_bucket.size() - 2;
						}
						_contains_launch(rank_id, buckets, bucket_length, int_result, indexes_buckets);
					}
				}

				if constexpr(DO_WORKLOAD_PROFILING) { workers_measures[worker_id] = omp_get_wtime(); }
				if constexpr(DO_TRACE) { __worker_done(); }

				// spdlog::info("Worker {} did {} launches", worker_id, done_container.size());
				
			}
			
			_pim_rankset.wait_all_ranks_done();

			// Formatting back into a vector of bool
            auto result =  std::vector<bool>();
			result.reserve(int_result.size());
            for (auto value : int_result) {
                result.emplace_back(value);
            }

			if constexpr(DO_WORKLOAD_PROFILING) {
				double stop = omp_get_wtime();
				for (size_t worker_id = 0; worker_id < nb_workers; worker_id++) {
					spdlog::info("Host worker {} had {} seconds of idle time", worker_id, stop - workers_measures[worker_id]);
				}
				_pim_rankset.end_workload_profiling();
			}

			if constexpr(DO_TRACE) { __method_end(); }

            return result;
		}

		size_t get_weight() override {

			if constexpr(DO_TRACE) { __method_start(); }

			uint64_t weight = 0;
			std::mutex weight_mutex;

			auto args = std::vector<uint64_t>{BloomFunction::BLOOM_WEIGHT};

			_pim_rankset.for_each_rank([&weight, &weight_mutex, this, &args](size_t rank_id) {
				_pim_rankset.broadcast_to_rank_async(rank_id, "args", 0, args);
				_pim_rankset.launch_rank_async(rank_id);
				_pim_rankset.add_callback_async(rank_id, [&weight, &weight_mutex, this, rank_id]() {
					auto result = _pim_rankset.get_reduced_sum_from_rank_sync<uint64_t>(rank_id, "result", 0, sizeof(uint64_t));
					weight_mutex.lock();
					weight += result;
					weight_mutex.unlock();
				});
			}, true); // Can execute in parallel

			_pim_rankset.wait_all_ranks_done();

			if constexpr(DO_TRACE) { __method_end(); }

			return static_cast<size_t>(weight);

		}

		const std::vector<uint8_t>& get_data() override {
			// NB: for ease we get the whole buffer no matter the actual filter size
			// because it would be a bit trickier to get only the relevant part used by each tasklet
			// So the data returned takes (a lot) more storage than it could
			// But works fine this way for now
			_bloom_data.resize(0);
			_bloom_data.reserve(TOTAL_MAX_BLOOM_DPU_SIZE * NR_TASKLETS * _pim_rankset.get_nb_dpu());
			size_t nb_elements = TOTAL_MAX_BLOOM_DPU_SIZE * NR_TASKLETS;
			size_t length = nb_elements * sizeof(uint8_t);
			_pim_rankset.for_each_rank([&nb_elements, &length, this](size_t rank_id) {
				auto rank_data = _pim_rankset.get_vec_data_from_rank_sync<uint8_t>(rank_id, "_bloom_data", 0, length);
				for (auto &dpu_data : rank_data) {
					// Cannot use dpu_data.end() iterator since size is empty (cf HACK in get_vec_data_from_rank_sync)
					std::move(dpu_data.begin(), dpu_data.begin() + nb_elements, std::back_inserter(_bloom_data)); 
				}
			}, false); // Sequential because we need the order to be deterministic
			return _bloom_data;
		}

		void set_data(const std::vector<uint8_t>& data) override {
			size_t start_index = 0;
			auto rank_buffers = std::vector<std::vector<std::vector<uint8_t>>>(_pim_rankset.get_nb_ranks());
			size_t nb_elements = TOTAL_MAX_BLOOM_DPU_SIZE * NR_TASKLETS;
			size_t length = nb_elements * sizeof(uint8_t);
			_pim_rankset.for_each_rank([this, &nb_elements, &length, &start_index, &data, &rank_buffers](size_t rank_id) {
				size_t nb_dpu_in_rank = _pim_rankset.get_nb_dpu_in_rank(rank_id);
				auto &buffers = rank_buffers[rank_id];
				buffers.resize(nb_dpu_in_rank);
				for (size_t dpu_id = 0; dpu_id < nb_dpu_in_rank; dpu_id++) {
					buffers[dpu_id].assign(data.begin() + start_index, data.begin() + start_index + nb_elements);
					start_index += nb_elements;
				}
				_pim_rankset.send_data_to_rank_async<uint8_t>(rank_id, "_bloom_data", 0, buffers , length);
			}, false); // Sequential because we need to restore in the same order it was got (but calls can be async)
			_pim_rankset.wait_all_ranks_done();
        }

	private:

		PimRankSet _pim_rankset;
		size_t _dpu_size2;
		HashPimItemDispatcher _item_dispatcher;
		std::vector<uint8_t> _bloom_data;

		const size_t _NO_MAPPING = UINT64_MAX;

		void _insert_launch(size_t rank_id, std::vector<std::vector<uint64_t>>& buckets, const size_t& bucket_length) {

			_pim_rankset.lock_rank(rank_id); // Lock so that other workers don't stack async calls in-between

			_pim_rankset.send_data_to_rank_async<uint64_t>(rank_id, "args", 0, buckets, bucket_length);
			_pim_rankset.add_callback_async(rank_id, [&buckets]() {
				buckets.clear(); // Slightly faster to clear memory as soon as possible
			});
			_pim_rankset.launch_rank_async(rank_id);

			_pim_rankset.unlock_rank(rank_id);

		}

		void _contains_launch(size_t rank_id, std::vector<std::vector<uint64_t>>& buckets, const size_t& bucket_length, std::vector<int>& lookup_results, std::vector<std::vector<size_t>>& indexes_buckets) {

			_pim_rankset.lock_rank(rank_id); // Lock so that other workers don't stack async calls in-between

			_pim_rankset.send_data_to_rank_async<uint64_t>(rank_id, "args", 0, buckets, bucket_length);
			_pim_rankset.add_callback_async(rank_id, [&buckets]() {
				buckets.clear(); // Slightly faster to clear memory as soon as possible
			});
			_pim_rankset.launch_rank_async(rank_id);
			_pim_rankset.add_callback_async(rank_id, [&indexes_buckets, bucket_length, &lookup_results, this, rank_id]() { // Get results
				// double start = omp_get_wtime();
				auto rank_results = _pim_rankset.get_vec_data_from_rank_sync<uint64_t>(rank_id, "args", 0, bucket_length);
				// double stop = omp_get_wtime();
				// spdlog::info("Transfer took {}", stop - start);
				// start = omp_get_wtime();
				for (size_t dpu_id = 0; dpu_id < indexes_buckets.size(); dpu_id++) {
					auto &indexes_bucket = indexes_buckets[dpu_id];
					auto &bucket_results = rank_results[dpu_id];
					for (size_t i = 0; i < bucket_results[0]; i++) {
						lookup_results[indexes_bucket[i]] = bucket_results[i + 2];
					}
				}
				indexes_buckets.clear(); // Slightly faster to clear memory as soon as possible
				// stop = omp_get_wtime();
				// spdlog::info("Writing results took {}", stop - start);
			});

			_pim_rankset.unlock_rank(rank_id);

		}

};