#ifndef F347470E_1730_41E9_9AE3_45A884CD2BFF
#define F347470E_1730_41E9_9AE3_45A884CD2BFF

#include "pim_common.h"
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
#include <memory>

// #define IGNORE_DPU_CALLS

void __attribute__((optimize(0))) _trace_rank_done() {}

enum DpuProfile {
    HARDWARE,
    SIMULATOR,
};


/* -------------------------------------------------------------------------- */
/*                        Management of a set of ranks                        */
/* -------------------------------------------------------------------------- */

class PimRankSet {

public:
    
    PimRankSet(size_t nb_ranks,
               size_t nb_threads = 8UL,
               DpuProfile dpu_profile = DpuProfile::HARDWARE,
               const char* binary_name = NULL) : _nb_ranks(nb_ranks), _nb_threads(nb_threads) {
        
        _sets.resize(_nb_ranks);
        _rank_mutexes = std::vector<std::mutex>(_nb_ranks);

        // Alloc in parallel
        #ifndef IGNORE_DPU_CALLS
        #pragma omp parallel for num_threads(_nb_threads)
		for (size_t rank_id = 0; rank_id < _nb_ranks; rank_id++) {
			DPU_ASSERT(dpu_alloc_ranks(1, _get_dpu_profile(dpu_profile).c_str(), &_sets[rank_id]));
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

    template<typename T>
    void broadcast_to_rank_sync(size_t rank_id, const char* symbol_name, uint32_t symbol_offset, const std::vector<T>& data) {
        #ifndef IGNORE_DPU_CALLS
        DPU_ASSERT(dpu_broadcast_to(_sets[rank_id], symbol_name, symbol_offset, data.data(), sizeof(T) * data.size(), DPU_XFER_DEFAULT));
        #endif
    }

    void broadcast_to_rank_async(size_t rank_id, const char* symbol_name, uint32_t symbol_offset, const void * src, size_t length) {
        #ifndef IGNORE_DPU_CALLS
        DPU_ASSERT(dpu_broadcast_to(_sets[rank_id], symbol_name, symbol_offset, src, length, DPU_XFER_ASYNC));
        #endif
    }

    template<typename T>
    void broadcast_to_rank_async(size_t rank_id, const char* symbol_name, uint32_t symbol_offset, const std::vector<T>& data) {
        #ifndef IGNORE_DPU_CALLS
        DPU_ASSERT(dpu_broadcast_to(_sets[rank_id], symbol_name, symbol_offset, data.data(), sizeof(T) * data.size(), DPU_XFER_ASYNC));
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

        #ifdef DO_DPU_PERFCOUNTER
        auto perf_value = get_reduced_sum_from_rank_sync<uint64_t>(rank_id, "perf_counter", 0, sizeof(uint64_t));
        auto perf_ref_id = get_reduced_sum_from_rank_sync<uint64_t>(rank_id, "perf_ref_id", 0, sizeof(uint64_t));
        spdlog::info("Average perf value is {} [{}]", perf_value / get_nb_dpu_in_rank(rank_id), perf_ref_id / get_nb_dpu_in_rank(rank_id));
        #endif
        
        #else
        std::this_thread::sleep_for(std::chrono::milliseconds(50)); // Simulate a 50ms run
        #endif
    }

    void launch_rank_async(size_t rank_id) {
        #ifndef IGNORE_DPU_CALLS
        if (_do_workload_profiling) {
            add_callback_async(rank_id, this, [](size_t rank_id, void* arg) {
                auto rankset = static_cast<PimRankSet*>(arg);
                double now = omp_get_wtime();
                rankset->_workload_measures[rank_id] += (now - rankset->_workload_begin[rank_id]);
            });
        }
        DPU_ASSERT(dpu_launch(_sets[rank_id], DPU_ASYNCHRONOUS));
        add_callback_async(rank_id, this, [](size_t rank_id, void* arg) {
            auto rankset = static_cast<PimRankSet*>(arg);
            rankset->_rank_finished_callback(rank_id);
        });
        #else
        _rank_finished_callback(rank_id);
        #endif
	}

    void start_workload_profiling() {
        double start = omp_get_wtime();
        _workload_measures.resize(0);
        _workload_measures.resize(_nb_ranks, 0.0);
        _workload_begin.resize(0);
        _workload_begin.resize(_nb_ranks, start);
        _do_workload_profiling = true;
    }

    void add_workload_profiling_callback(size_t rank_id) {
        if (_do_workload_profiling) {
            add_callback_async(rank_id, this, [](size_t rank_id, void* arg) {
                auto rankset = static_cast<PimRankSet*>(arg);
                double now = omp_get_wtime();
                rankset->_workload_begin[rank_id] = now;
            });
        }
    }

    void end_workload_profiling() {
        _do_workload_profiling = false;
        double stop = omp_get_wtime();
        for (size_t rank_id = 0; rank_id < _nb_ranks; rank_id++) {
            _workload_measures[rank_id] += (stop - _workload_begin[rank_id]);
            spdlog::info("Rank {} had {} seconds of idle time", rank_id, _workload_measures[rank_id]);
        }
    }

    void add_callback_async(size_t rank_id, void* arg, std::function<void (size_t, void*)> func) {
        auto callback_data = new CallbackData(rank_id, arg, func);
        DPU_ASSERT(dpu_callback(_sets[rank_id], PimRankSet::_generic_callback, callback_data, DPU_CALLBACK_ASYNC));
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
        struct dpu_set_t _it_dpu = dpu_set_t{};
	    uint32_t _it_dpu_idx = 0;
        DPU_FOREACH(_sets[rank_id], _it_dpu, _it_dpu_idx) {
            DPU_ASSERT(dpu_prepare_xfer(_it_dpu, &results[_it_dpu_idx]));
        }
		DPU_ASSERT(dpu_push_xfer(_sets[rank_id], DPU_XFER_FROM_DPU, symbol_name, symbol_offset, length, DPU_XFER_DEFAULT));
		for (size_t d = 0; d < _nb_dpu_in_rank[rank_id]; d++) {
			result += results[d];
        }
        #endif
        return result;
    }

    template<typename T>
    void send_data_to_rank_async(size_t rank_id, const char* symbol_name, uint32_t symbol_offset, std::vector<std::vector<T>>& buffers, size_t length) {
        #ifndef IGNORE_DPU_CALLS
        struct dpu_set_t _it_dpu = dpu_set_t{};
	    uint32_t _it_dpu_idx = 0;
        DPU_FOREACH(_sets[rank_id], _it_dpu, _it_dpu_idx) {
            DPU_ASSERT(dpu_prepare_xfer(_it_dpu, buffers[_it_dpu_idx].data()));
        }
        DPU_ASSERT(dpu_push_xfer(_sets[rank_id], DPU_XFER_TO_DPU, symbol_name, symbol_offset, length, DPU_XFER_ASYNC));
        #endif
    }

    template<typename T>
    void send_data_to_rank_sync(size_t rank_id, const char* symbol_name, uint32_t symbol_offset, std::vector<T> buffer, size_t length) {
        #ifndef IGNORE_DPU_CALLS
        struct dpu_set_t _it_dpu = dpu_set_t{};
	    uint32_t _it_dpu_idx = 0;
        DPU_FOREACH(_sets[rank_id], _it_dpu, _it_dpu_idx) {
            DPU_ASSERT(dpu_prepare_xfer(_it_dpu, &buffer[_it_dpu_idx]));
        }
        DPU_ASSERT(dpu_push_xfer(_sets[rank_id], DPU_XFER_TO_DPU, symbol_name, symbol_offset, length, DPU_XFER_DEFAULT));
        #endif
    }

    template<typename T>
    std::vector<std::vector<T>> get_vec_data_from_rank_sync(size_t rank_id, const char* symbol_name, uint32_t symbol_offset, size_t length) {
        auto buffer = std::vector<std::vector<T>>(get_nb_dpu_in_rank(rank_id));
        #ifndef IGNORE_DPU_CALLS
        struct dpu_set_t _it_dpu = dpu_set_t{};
	    uint32_t _it_dpu_idx = 0;
        DPU_FOREACH(_sets[rank_id], _it_dpu, _it_dpu_idx) {
            buffer[_it_dpu_idx].resize(length);
            DPU_ASSERT(dpu_prepare_xfer(_it_dpu, buffer[_it_dpu_idx].data()));
        }
        DPU_ASSERT(dpu_push_xfer(_sets[rank_id], DPU_XFER_FROM_DPU, symbol_name, symbol_offset, length, DPU_XFER_DEFAULT));
        #endif
        return buffer;
    }

private:
    
    std::vector<dpu_set_t> _sets;
    size_t _nb_ranks;
    size_t _nb_threads;

	size_t _nb_dpu;
	std::vector<size_t> _nb_dpu_in_rank;
	std::vector<size_t> _cum_dpu_idx_for_rank;

    std::vector<std::mutex> _rank_mutexes;

    std::string _get_dpu_profile(DpuProfile profile) {
        if (profile == DpuProfile::HARDWARE) {
            return "backend=hw";
        } else if (profile == DpuProfile::SIMULATOR) {
            return "backend=simulator";
        } else {
            return "";
        }
    }

    /* --------------------------- Workload profiling --------------------------- */

    bool _do_workload_profiling = false;
    std::vector<double> _workload_measures;
    std::vector<double> _workload_begin;

    /* -------------------------------- Callbacks ------------------------------- */

    class CallbackData {

        public:

            CallbackData(size_t rank_id, void* arg, std::function<void (size_t, void*)> func) : _rank_id(rank_id), _arg(arg), _func(func) {}
            void run() { _func(_rank_id, _arg); }
            size_t get_rank_id() { return _rank_id; }

        private:
        
            size_t _rank_id;
            void* _arg;
            std::function<void (size_t, void*)> _func;

        };

    static dpu_error_t _generic_callback([[maybe_unused]] struct dpu_set_t _set, [[maybe_unused]] uint32_t _id, void* arg) {
        auto callback_data = static_cast<CallbackData*>(arg);
        callback_data->run();
        delete callback_data;
        return DPU_OK;
    }

    void _rank_finished_callback(size_t rank_id) {
        // spdlog::debug("DPU callback: rank {} has finished", rank_id);
        _try_print_dpu_logs(rank_id);
        _trace_rank_done();
        
        #ifdef DO_DPU_PERFCOUNTER
        auto perf_value = get_reduced_sum_from_rank_sync<uint64_t>(rank_id, "perf_counter", 0, sizeof(uint64_t));
        auto perf_ref_id = get_reduced_sum_from_rank_sync<uint64_t>(rank_id, "perf_ref_id", 0, sizeof(uint64_t));
        // spdlog::info("Average perf value is {} [{}]", perf_value / get_nb_dpu_in_rank(rank_id), perf_ref_id / get_nb_dpu_in_rank(rank_id));
        #endif
	}

    /* --------------------------------- Logging -------------------------------- */

    bool _print_dpu_logs;

    void _try_print_dpu_logs(size_t rank_id) {
        #ifndef IGNORE_DPU_CALLS
        if (_print_dpu_logs) {
            struct dpu_set_t _it_dpu = dpu_set_t{};
            DPU_FOREACH(_sets[rank_id], _it_dpu) {
                DPU_ASSERT(dpu_log_read(_it_dpu, stdout));
            }
        }
        #endif
    }

};


/* -------------------------------------------------------------------------- */
/*                       Identifier of a DPU in the set                       */
/* -------------------------------------------------------------------------- */

class PimUnitUID {

    public:

        PimUnitUID(size_t rank_id, size_t dpu_id) : _rank_id(rank_id), _dpu_id(dpu_id) {}

        size_t get_rank_id() {
            return _rank_id;
        }

        size_t get_dpu_id() {
            return _dpu_id;
        }
    
    private:
        
        size_t _rank_id;
        size_t _dpu_id;

};


/* -------------------------------------------------------------------------- */
/*           Abstract class to dispatch something to a specific DPU           */
/* -------------------------------------------------------------------------- */

template <class T>
class PimDispatcher {

	public:

		PimDispatcher(PimRankSet& pim_rankset) : _pim_rankset(pim_rankset) {}
		
		virtual PimUnitUID dispatch(const T& arg) = 0;
	
	protected:

		PimRankSet& get_pim_rankset() {
			return _pim_rankset;
		}
		~PimDispatcher() = default;

	private:

		PimRankSet& _pim_rankset;

};


#endif /* F347470E_1730_41E9_9AE3_45A884CD2BFF */
