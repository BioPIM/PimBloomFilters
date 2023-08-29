#include "cxxopts/cxxopts.hpp"
#include "spdlog/spdlog.h"
#include <spdlog/sinks/basic_file_sink.h>

#include <stdint.h>
#include <stdio.h>
#include <iostream>
#include <random>
#include <algorithm>

#include "benchmark_utils.hpp"
#include "standard_bloom_filter.cpp"

constexpr size_t NB_THREADS = 8;
constexpr size_t NB_NO_ITEMS = 100000;

int main(int argc, char** argv) {

    cxxopts::Options options("benchmark2", "Run and time standard Bloom filters micro-benchmarks");

    options.add_options()
        ("k,hash", "Number of hash functions", cxxopts::value<size_t>()->default_value("8"))
        ("m,size2", "Size2 of the filter", cxxopts::value<size_t>()->default_value("20"))
        ("n,items", "Number of items", cxxopts::value<size_t>()->default_value("10000"))
        ("h,help", "Print usage")
        ("l,log", "Log perf values", cxxopts::value<bool>()->default_value("false"))
    ;

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    auto bench_logger = spdlog::basic_logger_mt("bench_logger", "bench2_perfs.csv");
    bench_logger->set_pattern("%v");

    size_t nb_hash = result["hash"].as<size_t>();
    size_t bloom_size2 = result["size2"].as<size_t>();
    size_t nb_items = result["items"].as<size_t>();

    bool do_log_perf = result["log"].as<bool>();
    auto log_timeit = [&bench_logger, nb_hash, bloom_size2, nb_items, do_log_perf](std::string id) {
        if (do_log_perf) {
            bench_logger->info("{}", get_last_timeit_log(id, nb_hash, bloom_size2, nb_items));
        }
    };

	std::vector<uint64_t> items = get_seq_items(nb_items);
    std::vector<uint64_t> no_items = get_seq_items(NB_NO_ITEMS, nb_items);

    spdlog::set_level(spdlog::level::info);

	std::cout << "> Creating filter..." << std::endl;
	std::unique_ptr<IBloomFilter> bloom_filter;
    TIMEIT(bloom_filter = std::make_unique<SyncCacheBloomFilter>(bloom_size2, nb_hash, NB_THREADS));
    log_timeit("init");

    std::cout << "> Inserting many items..." << std::endl;
	TIMEIT(bloom_filter->insert_bulk(items));
    log_timeit("insert");

    std::cout << "> Computing weight..." << std::endl;
    size_t weight;
    TIMEIT(weight = bloom_filter->get_weight());
    log_timeit("weight");
    std::cout << "Weight is " << weight << std::endl;

    std::cout << "> Querying all inserted items in a random order..." << std::endl;
	auto rng = std::default_random_engine{};
	std::shuffle(std::begin(items), std::end(items), rng);
	TIMEIT(bloom_filter->contains_bulk(items));
    log_timeit("lookup");

    std::cout << "> Querying non inserted items and checking fpr..." << std::endl;
    auto lookup_result = bloom_filter->contains_bulk(no_items);
	double fpr = (double) std::count(lookup_result.begin(), lookup_result.end(), true) / no_items.size();
    std::cout << "False positive rate is " << fpr << std::endl;

    // std::cout << "> Getting data..." << std::endl;
    // std::vector<uint8_t> data;
    // TIMEIT(data = bloom_filter->get_data());
    
    // std::cout << "> Creating a new filter..." << std::endl;
    // std::unique_ptr<IBloomFilter> bloom_filter2 = std::make_unique<SyncCacheBloomFilter>(bloom_size2, nb_hash, NB_THREADS);
    // std::cout << "Weight is " << bloom_filter2->get_weight() << std::endl;

    // std::cout << "> Loading data into the new filter..." << std::endl;
    // TIMEIT(bloom_filter2->set_data(data));

    // std::cout << "> Computing weight of new filter after data loading..." << std::endl;
    // std::cout << "Weight is " << bloom_filter2->get_weight() << std::endl;

    std::cout << "> The end." << std::endl;

}
