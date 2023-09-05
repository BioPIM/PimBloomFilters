#ifndef FD206EEF_A5B2_45D1_9607_FA5C11F4B995
#define FD206EEF_A5B2_45D1_9607_FA5C11F4B995

#include "spdlog/spdlog.h"

#include <vector>
#include <omp.h>
#include <cstdint>
#include <cstdio>
#include <sstream>

#include "bloom_filter.hpp"


/* -------------------------------------------------------------------------- */
/*                         Time measurement decorator                         */
/* -------------------------------------------------------------------------- */

template<typename T>
class BloomFilterTimeitDecorator : public IBloomFilter {

    public:

        template<typename... Args>
        BloomFilterTimeitDecorator(std::string log_params, std::string log_name, Args ... args) : _log_params(log_params) {
            static_assert(std::is_base_of<IBloomFilter, T>::value, "type parameter of this class must derive from IBloomFilter");
            _logger = spdlog::basic_logger_mt(log_name, log_name + ".csv");
            _logger->set_pattern("%v");
            _start_measure();
            _filter = std::make_unique<T>(args...);
            _stop_measure("init");
        }

        void insert(const uint64_t& item) override {
            _start_measure();
            _filter->insert(item);
            _stop_measure("insert");
        }

        void insert_bulk(const std::vector<uint64_t>& items) override {
            _start_measure();
            _filter->insert_bulk(items);
            _stop_measure("insert");
        }

        bool contains(const uint64_t& item) override {
            _start_measure();
            auto r = _filter->contains(item);
            _stop_measure("lookup");
            return r;
        }

        std::vector<bool> contains_bulk(const std::vector<uint64_t>& items) override {
            _start_measure();
            auto r = _filter->contains_bulk(items);
            _stop_measure("lookup");
            return r;
        }

        size_t get_weight() override {
            _start_measure();
            auto r = _filter->get_weight();
            _stop_measure("weight");
            return r;
        }

        const std::vector<uint8_t>& get_data() override {
            _start_measure();
            auto &r = _filter->get_data();
            _stop_measure("get_data");
            return r;
        }

        void set_data(const std::vector<uint8_t>& data) override {
            _start_measure();
            _filter->set_data(data);
            _stop_measure("set_data");
        }

        void log_data(double measure, std::string id) {
            if (!_log_params.empty()) {
                _logger->info("{},{},{}", measure, id, _log_params);
            }
        }

    private:

        std::unique_ptr<T> _filter;
        double _start_ts;
        std::string _log_params;
        std::shared_ptr<spdlog::logger> _logger;
        
        void _start_measure() {
            _start_ts = omp_get_wtime();
        }

        void _stop_measure(std::string id) {
            double stop_ts = omp_get_wtime();
            double measure = stop_ts - _start_ts;
            std::cout << "Took " << measure <<  " seconds" << std::endl;
            log_data(measure, id);
        }

};


/* -------------------------------------------------------------------------- */
/*                              Items generation                              */
/* -------------------------------------------------------------------------- */

std::vector<uint64_t> get_seq_items(const size_t nb, const uint64_t start_offset = 0) {
    std::vector<uint64_t> items;
    items.reserve(nb);
	for (size_t i = 0; i < nb; i++) {
		items.emplace_back(i + start_offset);
	}
    return items;
}


#endif /* FD206EEF_A5B2_45D1_9607_FA5C11F4B995 */
