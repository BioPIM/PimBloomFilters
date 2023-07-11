#pragma once

#include <dpu>
#include <vector>
#include <omp.h>
#include <functional>
#include <unistd.h>
#include <mutex>
#include <thread>
#include <utility>

class DpuProfile {
public:
    const static char* HARDWARE;
    const static char* SIMULATOR;
};

const char* DpuProfile::HARDWARE  = "backend=hw";
const char* DpuProfile::SIMULATOR = "backend=simulator";

class PimRankSet {

public:
    
    PimRankSet(int nb_ranks,
               int nb_threads = 8,
               const char* dpu_profile = DpuProfile::HARDWARE,
               const char* binary_name = NULL,
               bool print_dpu_logs = false,
               bool do_trace_debug = false) : _nb_ranks(nb_ranks), _nb_threads(nb_threads), _print_dpu_logs(print_dpu_logs), _do_trace_debug(do_trace_debug) {
        
        _sets = (dpu_set_t*) malloc(_nb_ranks * sizeof(dpu_set_t));

        // Alloc in parallel
        #pragma omp parallel for num_threads(_nb_threads)
		for (int rank_id = 0; rank_id < _nb_ranks; rank_id++) {
			DPU_ASSERT(dpu_alloc_ranks(1, dpu_profile, &_sets[rank_id]));
            if (binary_name != NULL) {
			    load_binary(binary_name, rank_id);
            }
		}

        // This part must be sequential
		_nb_dpu = 0;
        _nb_dpu_in_rank = std::vector<int>(_nb_ranks, 0);
        _cum_dpu_idx_for_rank = std::vector<int>(_nb_ranks, 0);
		for (int rank_id = 0; rank_id < _nb_ranks; rank_id++) {
			uint32_t nr_dpus;
			DPU_ASSERT(dpu_get_nr_dpus(_sets[rank_id], &nr_dpus));
            _cum_dpu_idx_for_rank[rank_id] = _nb_dpu; // Set before add to have starting idx
			_nb_dpu += nr_dpus;
            _nb_dpu_in_rank[rank_id] = nr_dpus;
		}

        _is_rank_idle = std::vector<bool>(_nb_ranks, true);

        if (do_trace_debug) {
            _trace_debug_status = std::vector<int>(_nb_ranks, TraceDebugStatus::NONE);
            for (int rank_id = 0; rank_id < _nb_ranks; rank_id++) {
                _trace_debug_mutex.push_back(new std::mutex());
                std::thread t(&PimRankSet::_watch_trace_debug, this, rank_id);
                t.detach();
            }
        }

    }

    ~PimRankSet() {
        #pragma omp parallel for num_threads(_nb_threads)
        for (int rank_id = 0; rank_id < _nb_ranks; rank_id++) {
            DPU_ASSERT(dpu_free(_sets[rank_id]));
        }
        free(_sets);
        if (_do_trace_debug) {
            for (int rank_id = 0; rank_id < _nb_ranks; rank_id++) {
                _update_trace_debug_status(rank_id, TraceDebugStatus::DONE);
            }
        }
    }

    int get_nb_dpu() { return _nb_dpu; }
    int get_nb_ranks() { return _nb_ranks; }
    int get_nb_dpu_in_rank(int rank_id) { return _nb_dpu_in_rank[rank_id]; }

    void load_binary(const char* binary_name, int rank) {
        // TODO: check if binary exists
        DPU_ASSERT(dpu_load(_sets[rank], binary_name, NULL));
    }

    void broadcast_to_rank(int rank_id, const char* symbol_name, uint32_t symbol_offset, const void * src, size_t length) {
        DPU_ASSERT(dpu_broadcast_to(_sets[rank_id], symbol_name, symbol_offset, src, length, DPU_XFER_DEFAULT));
    }

    void for_each_rank(std::function<void (int)> lambda, bool can_parallel = false) {
        #pragma omp parallel for num_threads(_nb_threads) if(can_parallel)
        for (int rank_id = 0; rank_id < _nb_ranks; rank_id++) {
            lambda(rank_id);
        }
    }

    void launch_rank_sync(int rank_id) {
        wait_rank_ready(rank_id);
        _update_idle_status(rank_id, false);
        DPU_ASSERT(dpu_launch(_sets[rank_id], DPU_SYNCHRONOUS));
        _try_print_dpu_logs(rank_id);
        _update_idle_status(rank_id, true);
    }

    void launch_rank_async(int rank_id) {
        wait_rank_ready(rank_id);
		DPU_ASSERT(dpu_launch(_sets[rank_id], DPU_ASYNCHRONOUS));
		dpu_callback(_sets[rank_id], PimRankSet::_rank_done_callback, new std::pair<PimRankSet*, int>(this, rank_id), DPU_CALLBACK_ASYNC);
		_update_idle_status(rank_id, false);
        if (_do_trace_debug) {
           _update_trace_debug_status(rank_id, TraceDebugStatus::WATCH);
        }
	}

    void wait_rank_ready(int rank_id) {
        while (!_is_rank_idle[rank_id]) { usleep(100); }
    }

    bool is_rank_ready(int rank_id) {
        return _is_rank_idle[rank_id];
    }

    template<typename T>
    T get_reduced_sum_from_rank(int rank_id, const char* symbol_name, uint32_t symbol_offset, size_t length) {
        T result = 0;
        T results[_nb_dpu_in_rank[rank_id]];
        DPU_FOREACH(_sets[rank_id], _it_dpu, _it_dpu_idx) {
            DPU_ASSERT(dpu_prepare_xfer(_it_dpu, &results[_it_dpu_idx]));
        }
		DPU_ASSERT(dpu_push_xfer(_sets[rank_id], DPU_XFER_FROM_DPU, symbol_name, symbol_offset, length, DPU_XFER_DEFAULT));
		for (int d = 0; d < _nb_dpu_in_rank[rank_id]; d++) {
			result += results[d];
        }
        return result;
    }

    template<typename T>
    void send_data_to_rank(int rank_id, const char* symbol_name, uint32_t symbol_offset, const std::vector<T*>& buffers, size_t length) {
        DPU_FOREACH(_sets[rank_id], _it_dpu, _it_dpu_idx) {
            DPU_ASSERT(dpu_prepare_xfer(_it_dpu, buffers[_it_dpu_idx]));
        }
        DPU_ASSERT(dpu_push_xfer(_sets[rank_id], DPU_XFER_TO_DPU, symbol_name, symbol_offset, length, DPU_XFER_DEFAULT));
    }

private:

    dpu_set_t* _sets;
    int _nb_ranks;

	int _nb_dpu;
	std::vector<int> _nb_dpu_in_rank;
	std::vector<int> _cum_dpu_idx_for_rank;

    std::mutex idle_mutex;
    std::vector<bool> _is_rank_idle;

    void _update_idle_status(int rank_id, bool value) {
        idle_mutex.lock();
        _is_rank_idle[rank_id] = value;
        idle_mutex.unlock();
    }

    static dpu_error_t _rank_done_callback(struct dpu_set_t set, uint32_t _id, void* arg) {
		std::pair<PimRankSet*, int>* info = (std::pair<PimRankSet*, int>*) arg;
		// std::cout << "Rank " << info->second << " is done" << std::endl;
		info->first->_update_idle_status(info->second, true);
        info->first->_try_print_dpu_logs(info->second);
		delete info;
		return DPU_OK;
	}

    int _nb_threads;

    bool _print_dpu_logs;

    void _try_print_dpu_logs(int rank_id) {
        if (_print_dpu_logs) {
            DPU_FOREACH(_sets[rank_id], _it_dpu) {
                DPU_ASSERT(dpu_log_read(_it_dpu, stdout));
            }
        }
    }
	
    // For iterations
	struct dpu_set_t _it_dpu;
	uint32_t _it_dpu_idx;

    // Trace debug info
    bool _do_trace_debug;
    std::vector<int> _trace_debug_status;
	std::vector<std::mutex*> _trace_debug_mutex;
    enum TraceDebugStatus{NONE, WATCH, DONE};

    void _watch_trace_debug(int rank_id) {
		while (_trace_debug_status[rank_id] != TraceDebugStatus::DONE) {
			if (_trace_debug_status[rank_id] == TraceDebugStatus::WATCH) {
				_update_trace_debug_status(rank_id, TraceDebugStatus::NONE);
				DPU_ASSERT(dpu_sync(_sets[rank_id]));
			}
			usleep(100);
		}
        delete _trace_debug_mutex[rank_id];
	}

    void _update_trace_debug_status(int rank_id, TraceDebugStatus value) {
        _trace_debug_mutex[rank_id]->lock();
        if (_trace_debug_status[rank_id] != TraceDebugStatus::DONE) {
            _trace_debug_status[rank_id] = value;
        }
        _trace_debug_mutex[rank_id]->unlock();
    }

};