#ifndef F347470E_1730_41E9_9AE3_45A884CD2BFF
#define F347470E_1730_41E9_9AE3_45A884CD2BFF

#include "pim_common.h"
#include "pim_api.hpp"
#include "spdlog/spdlog.h"

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

// #define DO_WORKLOAD_PROFILING
// #define DO_TRACE

void __attribute__((optimize(0))) _trace_rank_done() {}

/* -------------------------------------------------------------------------- */
/*                                 DPU profile                                */
/* -------------------------------------------------------------------------- */

class DpuProfile {

public:

    enum Backend {
        HARDWARE, // Default
        SIMULATOR,
    };

    DpuProfile() : _backend(Backend::HARDWARE) {}

    DpuProfile& set_backend(Backend value) {
        _backend = value;
        return *this;
    }

    std::string get() {
        std::string profile = "backend=";
        if (_backend == Backend::HARDWARE) {
            profile += "hw";
        } else if (_backend == Backend::HARDWARE) {
            profile += "simulator";
        }
        return profile;
    }

private:

    Backend _backend;


};


/* -------------------------------------------------------------------------- */
/*                        Management of a set of ranks                        */
/* -------------------------------------------------------------------------- */

class PimRankSet {

public:
    
    PimRankSet(size_t nb_ranks,
               size_t nb_threads = 8UL,
               DpuProfile dpu_profile = DpuProfile(),
               const std::string binary_name = "") : _nb_ranks(nb_ranks), _nb_threads(nb_threads) {
        
        _sets.resize(_nb_ranks);
        _rank_mutexes = std::vector<std::mutex>(_nb_ranks);

        // Alloc in parallel
        std::string profile = dpu_profile.get();
        #pragma omp parallel for num_threads(_nb_threads)
		for (size_t rank_id = 0; rank_id < _nb_ranks; rank_id++) {
			_pim_api.dpu_alloc_ranks(1, profile.c_str(), &_sets[rank_id]);
            if (!binary_name.empty()) {
			    load_binary(binary_name.c_str(), rank_id);
            }
		}

        // This part must be sequential
		_nb_dpu = 0;
        _nb_dpu_in_rank.resize(_nb_ranks, 0);
		for (size_t rank_id = 0; rank_id < _nb_ranks; rank_id++) {
			uint32_t nr_dpus = 0;
			_pim_api.dpu_get_nr_dpus(_sets[rank_id], &nr_dpus);
			_nb_dpu += nr_dpus;
            _nb_dpu_in_rank[rank_id] = nr_dpus;
		}

    }

    ~PimRankSet() {
        #pragma omp parallel for num_threads(_nb_threads)
        for (size_t rank_id = 0; rank_id < _nb_ranks; rank_id++) {
            _pim_api.dpu_free(_sets[rank_id]);
        }
    }

    /* -------------------------- Get count information ------------------------- */

    size_t get_nb_dpu() { return _nb_dpu; }
    size_t get_nb_ranks() { return _nb_ranks; }
    size_t get_nb_dpu_in_rank(size_t rank_id) { return _nb_dpu_in_rank[rank_id]; }

    /* -------------------------------- Iterating ------------------------------- */

    void for_each_rank(std::function<void (size_t)> lambda, bool can_parallel = false) {
        #pragma omp parallel for num_threads(_nb_threads) if(can_parallel)
        for (size_t rank_id = 0; rank_id < _nb_ranks; rank_id++) {
            lambda(rank_id);
        }
    }

    /* ------------------------------- Load binary ------------------------------ */

    void load_binary(const char* binary_name, size_t rank) {
        if (std::filesystem::exists(binary_name)) {
            _pim_api.dpu_load(_sets[rank], binary_name, NULL);
        } else {
            spdlog::critical("DPU binary program at {} does not exist", binary_name);
        }
    }

    /* ----------------------------- Broadcasts sync ---------------------------- */

    void broadcast_to_rank_sync(size_t rank_id, const char* symbol_name, uint32_t symbol_offset, const void * src, size_t length) {
        _pim_api.dpu_broadcast_to(_sets[rank_id], symbol_name, symbol_offset, src, length, DPU_XFER_DEFAULT);
    }

    template<typename T>
    void broadcast_to_rank_sync(size_t rank_id, const char* symbol_name, uint32_t symbol_offset, std::vector<T>& data) {
        _pim_api.dpu_broadcast_to(_sets[rank_id], symbol_name, symbol_offset, data.data(), sizeof(T) * data.size(), DPU_XFER_DEFAULT);
    }

    /* ---------------------------- Broadcasts async ---------------------------- */

    void broadcast_to_rank_async(size_t rank_id, const char* symbol_name, uint32_t symbol_offset, const void * src, size_t length) {
        _pim_api.dpu_broadcast_to(_sets[rank_id], symbol_name, symbol_offset, src, length, DPU_XFER_ASYNC);
    }

    template<typename T>
    void broadcast_to_rank_async(size_t rank_id, const char* symbol_name, uint32_t symbol_offset, const std::vector<T>& data) {
        _pim_api.dpu_broadcast_to(_sets[rank_id], symbol_name, symbol_offset, data.data(), sizeof(T) * data.size(), DPU_XFER_ASYNC);
    }

    /* ------------------------------- Launch sync ------------------------------ */

    void launch_rank_sync(size_t rank_id) {
        _pim_api.dpu_launch(_sets[rank_id], DPU_SYNCHRONOUS);
        wait_rank_done(rank_id);

        #ifdef DO_TRACE
        _trace_rank_done();
        #endif

        #ifdef LOG_DPU
        _print_dpu_logs(rank_id);
        #endif

        #ifdef DO_DPU_PERFCOUNTER
        _log_perfcounter(rank_id);
        #endif
    }

    /* ------------------------------ Launch async ------------------------------ */

    void launch_rank_async(size_t rank_id) {
        _pim_api.dpu_launch(_sets[rank_id], DPU_ASYNCHRONOUS);

        #ifdef DO_TRACE
        add_callback_async(rank_id, []() {
            _trace_rank_done();
        });
        #endif

        #ifdef LOG_DPU
        add_callback_async(rank_id, [this, rank_id]() {
            _print_dpu_logs(rank_id);
        });
        #endif

        #ifdef DO_DPU_PERFCOUNTER
        add_callback_async(rank_id, [this, rank_id]() {
            _log_perfcounter(rank_id);
        });
        #endif
	}

    /* -------------------------------- Callbacks ------------------------------- */

    void add_callback_async(size_t rank_id, std::function<void (void)> func) {
        auto callback_func = new std::function<void (void)>(func);
        _pim_api.dpu_callback(_sets[rank_id], PimRankSet::_generic_callback, (void *)callback_func, DPU_CALLBACK_ASYNC);
    }

    /* ------------------------------ Rank locking ------------------------------ */

    void lock_rank(size_t rank_id) {
        _rank_mutexes[rank_id].lock();
        if (_do_workload_profiling) {
            add_callback_async(rank_id, [this, rank_id]() {
                double now = omp_get_wtime();
                _workload_measures[rank_id] += (now - _workload_begin[rank_id]);
            });
        }
    }

    void unlock_rank(size_t rank_id) {
        if (_do_workload_profiling) {
            add_callback_async(rank_id, [this, rank_id]() {
                double now = omp_get_wtime();
                _workload_begin[rank_id] = now;
            });
        }
        _rank_mutexes[rank_id].unlock();
    }

    /* ------------------------------ Waiting done ------------------------------ */

    void wait_rank_done(size_t rank_id) {
        _pim_api.dpu_sync(_sets[rank_id]);
    }

    void wait_all_ranks_done() {
        #pragma omp parallel for num_threads(_nb_threads)
        for (size_t rank_id = 0; rank_id < _nb_ranks; rank_id++) {
            wait_rank_done(rank_id);
        }
    }

    /* ----------------------------- Send data sync ----------------------------- */

    template<typename T>
    void send_data_to_rank_sync(size_t rank_id, const char* symbol_name, uint32_t symbol_offset, std::vector<T> buffer, size_t length) {
        struct dpu_set_t _it_dpu = dpu_set_t{};
	    uint32_t _it_dpu_idx = 0;
        DPU_FOREACH(_sets[rank_id], _it_dpu, _it_dpu_idx) {
            _pim_api.dpu_prepare_xfer(_it_dpu, &buffer[_it_dpu_idx]);
        }
        _pim_api.dpu_push_xfer(_sets[rank_id], DPU_XFER_TO_DPU, symbol_name, symbol_offset, length, DPU_XFER_DEFAULT);
    }

    /* ----------------------------- Send data async ---------------------------- */

    template<typename T>
    void send_data_to_rank_async(size_t rank_id, const char* symbol_name, uint32_t symbol_offset, std::vector<std::vector<T>>& buffers, size_t length) {
        struct dpu_set_t _it_dpu = dpu_set_t{};
	    uint32_t _it_dpu_idx = 0;
        DPU_FOREACH(_sets[rank_id], _it_dpu, _it_dpu_idx) {
            _pim_api.dpu_prepare_xfer(_it_dpu, buffers[_it_dpu_idx].data());
        }
        _pim_api.dpu_push_xfer(_sets[rank_id], DPU_XFER_TO_DPU, symbol_name, symbol_offset, length, DPU_XFER_ASYNC);
    }

    /* --------------------------- Retrieve data sync --------------------------- */

    template<typename T>
    T get_reduced_sum_from_rank_sync(size_t rank_id, const char* symbol_name, uint32_t symbol_offset, size_t length) {
        auto results = std::vector<T>(get_nb_dpu_in_rank(rank_id));
        struct dpu_set_t _it_dpu = dpu_set_t{};
	    uint32_t _it_dpu_idx = 0;
        DPU_FOREACH(_sets[rank_id], _it_dpu, _it_dpu_idx) {
            _pim_api.dpu_prepare_xfer(_it_dpu, &results[_it_dpu_idx]);
        }
		_pim_api.dpu_push_xfer(_sets[rank_id], DPU_XFER_FROM_DPU, symbol_name, symbol_offset, length, DPU_XFER_DEFAULT);
		T result = 0;
        for (size_t d = 0; d < _nb_dpu_in_rank[rank_id]; d++) {
			result += results[d];
        }
        return result;
    }

    template<typename T>
    std::vector<std::vector<T>> get_vec_data_from_rank_sync(size_t rank_id, const char* symbol_name, uint32_t symbol_offset, size_t length) {
        auto buffer = std::vector<std::vector<T>>(get_nb_dpu_in_rank(rank_id));
        struct dpu_set_t _it_dpu = dpu_set_t{};
	    uint32_t _it_dpu_idx = 0;
        DPU_FOREACH(_sets[rank_id], _it_dpu, _it_dpu_idx) {
            /* ------------------------------- BEGIN HACK ------------------------------- */
            // This is much faster to reserve than resize because nothing is initialized
            // The transfer will set the data
            // BUT the vectors are "officially" empty so cannot iterate on it or use size()
            // Can access with [] but be careful with the index!
            buffer[_it_dpu_idx].reserve(length);
            /* -------------------------------- END HACK -------------------------------- */
            _pim_api.dpu_prepare_xfer(_it_dpu, buffer[_it_dpu_idx].data());
        }
        _pim_api.dpu_push_xfer(_sets[rank_id], DPU_XFER_FROM_DPU, symbol_name, symbol_offset, length, DPU_XFER_DEFAULT);
        return buffer;
    }

    /* ---------------------------- Profiling methods --------------------------- */

    void start_workload_profiling() {
        #ifdef DO_WORKLOAD_PROFILING
        double start = omp_get_wtime();
        _workload_measures.resize(0);
        _workload_measures.resize(_nb_ranks, 0.0);
        _workload_begin.resize(0);
        _workload_begin.resize(_nb_ranks, start);
        _do_workload_profiling = true;
        #endif
    }

    void end_workload_profiling() {
        if (_do_workload_profiling) {
            _do_workload_profiling = false;
            for (size_t rank_id = 0; rank_id < _nb_ranks; rank_id++) {
                spdlog::info("Rank {} had {} seconds of idle time", rank_id, _workload_measures[rank_id]);
            }
        }
    }

    /* ------------------------------ Debug helpers ----------------------------- */

    void broadcast_dpu_uid() {
        for_each_rank([this](size_t rank_id) {
            size_t nb_dpus_in_rank = get_nb_dpu_in_rank(rank_id);
            auto uids = std::vector<size_t>(nb_dpus_in_rank, 0);
            for (size_t i = 0; i < nb_dpus_in_rank; i++) {
            	uids[i] = _get_dpu_uid(rank_id, i);
            }
            send_data_to_rank_sync(rank_id, "dpu_uid", 0, uids, sizeof(size_t));
        });
    }

private:
    
    ProxyPimAPI _pim_api; // Use DummyPimAPI instead of ProxyPimAPI to ignore the dpu lib
    std::vector<dpu_set_t> _sets;
    size_t _nb_ranks;
    size_t _nb_threads;

	size_t _nb_dpu;
	std::vector<size_t> _nb_dpu_in_rank;

    std::vector<std::mutex> _rank_mutexes;

    static inline size_t _get_dpu_uid(size_t rank_id, size_t dpu_id) {
        return rank_id * 100 + dpu_id;
    }

    /* --------------------------- Workload profiling --------------------------- */

    bool _do_workload_profiling = false;
    std::vector<double> _workload_measures;
    std::vector<double> _workload_begin;

    /* -------------------------------- Callbacks ------------------------------- */

    static dpu_error_t _generic_callback([[maybe_unused]] struct dpu_set_t _set, [[maybe_unused]] uint32_t _id, void* arg) {
        auto func = static_cast<std::function<void (void)>*>(arg);
        (*func)();
        delete func;
        return DPU_OK;
    }

    /* --------------------------------- Logging -------------------------------- */

    void _print_dpu_logs(size_t rank_id) {
        struct dpu_set_t _it_dpu = dpu_set_t{};
        DPU_FOREACH(_sets[rank_id], _it_dpu) {
            _pim_api.dpu_log_read(_it_dpu, stdout);
        }
    }

    void _log_perfcounter(size_t rank_id) {
        auto perf_value = get_reduced_sum_from_rank_sync<uint64_t>(rank_id, "perf_counter", 0, sizeof(uint64_t));
        auto perf_ref_id = get_reduced_sum_from_rank_sync<uint64_t>(rank_id, "perf_ref_id", 0, sizeof(uint64_t));
        spdlog::info("Average perf value is {} [{}]", perf_value / get_nb_dpu_in_rank(rank_id), perf_ref_id / get_nb_dpu_in_rank(rank_id));
    }

};


/* -------------------------------------------------------------------------- */
/*                       Identifier of a DPU in the set                       */
/* -------------------------------------------------------------------------- */

class PimUnitUID {

    public:

        PimUnitUID() : _rank_id(0), _dpu_id(0) {}
        PimUnitUID(size_t rank_id, size_t dpu_id) : _rank_id(rank_id), _dpu_id(dpu_id) {}

        PimUnitUID& operator=(const PimUnitUID& that) {
            _rank_id = that._rank_id;
            _dpu_id = that._dpu_id;
            return *this;
        }

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
