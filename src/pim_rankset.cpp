#pragma once

#include "spdlog/spdlog.h"

#include <dpu>
#include <vector>
#include <omp.h>
#include <functional>
#include <unistd.h>
#include <mutex>
#include <utility>
#include <filesystem>
#include <chrono>
#include <thread>

// #define IGNORE_DPU_CALLS

void _trace_rank_done() {}

class DpuProfile {
public:
    const static char* HARDWARE;
    const static char* SIMULATOR;
};

const char* DpuProfile::HARDWARE  = "backend=hw";
const char* DpuProfile::SIMULATOR = "backend=simulator";

class PimCallbackData {

public:

    PimCallbackData(size_t rank_id, void* arg, std::function<void (size_t, void*)> func) : _rank_id(rank_id), _arg(arg), _func(func) {}
    void run() { _func(_rank_id, _arg); }
    size_t get_rank_id() { return _rank_id; }

private:
    size_t _rank_id;
    void* _arg;
    std::function<void (size_t, void*)> _func;

};

class PimRankSet {

public:
    
    PimRankSet(size_t nb_ranks,
               size_t nb_threads = 8UL,
               const char* dpu_profile = DpuProfile::HARDWARE,
               const char* binary_name = NULL) : _nb_ranks(nb_ranks), _nb_threads(nb_threads) {
        
        _sets.resize(_nb_ranks);
        _rank_mutexes = std::vector<std::mutex>(_nb_ranks);

        // Alloc in parallel
        #ifndef IGNORE_DPU_CALLS
        #pragma omp parallel for num_threads(_nb_threads)
		for (size_t rank_id = 0; rank_id < _nb_ranks; rank_id++) {
			DPU_ASSERT(dpu_alloc_ranks(1, dpu_profile, &_sets[rank_id]));
            if (binary_name != NULL) {
			    load_binary(binary_name, rank_id);
            }
		}
        #endif

        // This part must be sequential
		_nb_dpu = 0;
        _nb_dpu_in_rank = std::vector<size_t>(_nb_ranks, 0);
        _cum_dpu_idx_for_rank = std::vector<size_t>(_nb_ranks, 0);
		for (size_t rank_id = 0; rank_id < _nb_ranks; rank_id++) {
			uint32_t nr_dpus = 0;
            #ifndef IGNORE_DPU_CALLS
			DPU_ASSERT(dpu_get_nr_dpus(_sets[rank_id], &nr_dpus));
            #else
            nr_dpus = 64; // Let's simulate 64 DPUs per rank
            #endif
            _cum_dpu_idx_for_rank[rank_id] = _nb_dpu; // Set before add to have starting idx
			_nb_dpu += nr_dpus;
            _nb_dpu_in_rank[rank_id] = nr_dpus;
		}


        #ifdef LOG_DPU
        _print_dpu_logs = true;
        #else
        _print_dpu_logs = false;
        #endif

    }

    ~PimRankSet() {
        #ifndef IGNORE_DPU_CALLS
        #pragma omp parallel for num_threads(_nb_threads)
        for (size_t rank_id = 0; rank_id < _nb_ranks; rank_id++) {
            DPU_ASSERT(dpu_free(_sets[rank_id]));
        }
        #endif
    }

    size_t get_nb_dpu() { return _nb_dpu; }
    size_t get_nb_ranks() { return _nb_ranks; }
    size_t get_nb_dpu_in_rank(size_t rank_id) { return _nb_dpu_in_rank[rank_id]; }
    size_t get_cum_dpu_idx_for_rank(size_t rank_id) { return _cum_dpu_idx_for_rank[rank_id]; }

    void load_binary(const char* binary_name, size_t rank) {
        if (std::filesystem::exists(binary_name)) {
            #ifndef IGNORE_DPU_CALLS
            DPU_ASSERT(dpu_load(_sets[rank], binary_name, NULL));
            #endif
        } else {
            spdlog::critical("DPU binary program at {} does not exist", binary_name);
        }
    }

    void broadcast_to_rank_sync(size_t rank_id, const char* symbol_name, uint32_t symbol_offset, const void * src, size_t length) {
        #ifndef IGNORE_DPU_CALLS
        DPU_ASSERT(dpu_broadcast_to(_sets[rank_id], symbol_name, symbol_offset, src, length, DPU_XFER_DEFAULT));
        #endif
    }

    void broadcast_to_rank_async(size_t rank_id, const char* symbol_name, uint32_t symbol_offset, const void * src, size_t length) {
        #ifndef IGNORE_DPU_CALLS
        DPU_ASSERT(dpu_broadcast_to(_sets[rank_id], symbol_name, symbol_offset, src, length, DPU_XFER_ASYNC));
        #endif
    }

    void for_each_rank(std::function<void (size_t)> lambda, bool can_parallel = false) {
        #pragma omp parallel for num_threads(_nb_threads) if(can_parallel)
        for (size_t rank_id = 0; rank_id < _nb_ranks; rank_id++) {
            lambda(rank_id);
        }
    }

    void launch_rank_sync(size_t rank_id) {
        #ifndef IGNORE_DPU_CALLS
        DPU_ASSERT(dpu_launch(_sets[rank_id], DPU_SYNCHRONOUS));
        wait_rank_done(rank_id);
        _try_print_dpu_logs(rank_id);
        #else
        std::this_thread::sleep_for(std::chrono::milliseconds(50)); // Simulate a 50ms run
        #endif
    }

    void launch_rank_async(size_t rank_id) {
        #ifndef IGNORE_DPU_CALLS
        DPU_ASSERT(dpu_launch(_sets[rank_id], DPU_ASYNCHRONOUS));
        auto callback_data = new PimCallbackData(rank_id, this, [](size_t rank_id, void* arg) {
            auto rankset = static_cast<PimRankSet*>(arg);
            rankset->_rank_finished_callback(rank_id);
        });
        add_callback_async(callback_data);
        #else
        _rank_finished_callback(rank_id);
        #endif
	}

    void add_callback_async(PimCallbackData* callback_data) {
        DPU_ASSERT(dpu_callback(_sets[callback_data->get_rank_id()], PimRankSet::_generic_callback, callback_data, DPU_CALLBACK_ASYNC));
    }

    void lock_rank(size_t rank_id) {
        _rank_mutexes[rank_id].lock();
    }

     void unlock_rank(size_t rank_id) {
        _rank_mutexes[rank_id].unlock();
    }

    void wait_rank_done(size_t rank_id) {
        #ifndef IGNORE_DPU_CALLS
        DPU_ASSERT(dpu_sync(_sets[rank_id]));
        #endif
    }

    void wait_all_ranks_done() {
        #pragma omp parallel for num_threads(_nb_threads)
        for (size_t rank_id = 0; rank_id < _nb_ranks; rank_id++) {
            wait_rank_done(rank_id);
        }
    }

    template<typename T>
    T get_reduced_sum_from_rank_sync(size_t rank_id, const char* symbol_name, uint32_t symbol_offset, size_t length) {
        T result = 0;
        #ifndef IGNORE_DPU_CALLS
        auto results = std::vector<T>(_nb_dpu_in_rank[rank_id]);
        struct dpu_set_t _it_dpu;
	    uint32_t _it_dpu_idx;
        DPU_FOREACH(_sets[rank_id], _it_dpu, _it_dpu_idx) {
            DPU_ASSERT(dpu_prepare_xfer(_it_dpu, &results[_it_dpu_idx]));
        }
		DPU_ASSERT(dpu_push_xfer(_sets[rank_id], DPU_XFER_FROM_DPU, symbol_name, symbol_offset, length, DPU_XFER_DEFAULT));
        std::string sizes = std::string();
		for (size_t d = 0; d < _nb_dpu_in_rank[rank_id]; d++) {
			result += results[d];
            sizes += std::to_string(results[d]) + std::string(" ");
        }
        spdlog::debug("Rank {}: reduced result = {} [ {} ]", rank_id, result, sizes);
        #endif
        return result;
    }

    template<typename T>
    void send_data_to_rank_async(size_t rank_id, const char* symbol_name, uint32_t symbol_offset, std::vector<std::vector<T>>& buffers, size_t length) {
        uint64_t items_sent = 0;
        #ifndef IGNORE_DPU_CALLS
        struct dpu_set_t _it_dpu;
	    uint32_t _it_dpu_idx;
        DPU_FOREACH(_sets[rank_id], _it_dpu, _it_dpu_idx) {
            DPU_ASSERT(dpu_prepare_xfer(_it_dpu, buffers[_it_dpu_idx].data()));
            items_sent += buffers[_it_dpu_idx][0];
        }
        DPU_ASSERT(dpu_push_xfer(_sets[rank_id], DPU_XFER_TO_DPU, symbol_name, symbol_offset, length, DPU_XFER_ASYNC));
        spdlog::debug("Rank {}: Sending {} items", rank_id, items_sent);
        #endif
    }

    template<typename T>
    void send_data_to_rank_sync(size_t rank_id, const char* symbol_name, uint32_t symbol_offset, std::vector<T> buffer, size_t length) {
        #ifndef IGNORE_DPU_CALLS
        struct dpu_set_t _it_dpu;
	    uint32_t _it_dpu_idx;
        DPU_FOREACH(_sets[rank_id], _it_dpu, _it_dpu_idx) {
            DPU_ASSERT(dpu_prepare_xfer(_it_dpu, &buffer[_it_dpu_idx]));
        }
        DPU_ASSERT(dpu_push_xfer(_sets[rank_id], DPU_XFER_TO_DPU, symbol_name, symbol_offset, length, DPU_XFER_DEFAULT));
        #endif
    }

private:
    
    std::vector<dpu_set_t> _sets;
    size_t _nb_ranks;
    size_t _nb_threads;

	size_t _nb_dpu;
	std::vector<size_t> _nb_dpu_in_rank;
	std::vector<size_t> _cum_dpu_idx_for_rank;

    std::vector<std::mutex> _rank_mutexes;

    // Callbacks
    static dpu_error_t _generic_callback([[maybe_unused]] struct dpu_set_t _set, [[maybe_unused]] uint32_t _id, void* arg) {
        auto callback_data = static_cast<PimCallbackData*>(arg);
        callback_data->run();
        delete callback_data;
        return DPU_OK;
    }

    void _rank_finished_callback(size_t rank_id) {
        spdlog::info("DPU callback: rank {} has finished", rank_id);
        _try_print_dpu_logs(rank_id);
        _trace_rank_done();
	}

    // Logging
    bool _print_dpu_logs;

    void _try_print_dpu_logs(size_t rank_id) {
        #ifndef IGNORE_DPU_CALLS
        if (_print_dpu_logs) {
            struct dpu_set_t _it_dpu;
            DPU_FOREACH(_sets[rank_id], _it_dpu) {
                DPU_ASSERT(dpu_log_read(_it_dpu, stdout));
            }
        }
        #endif
    }

};