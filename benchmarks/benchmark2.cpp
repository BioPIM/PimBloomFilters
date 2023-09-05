#include "cxxopts/cxxopts.hpp"
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

    size_t nb_hash = result["hash"].as<size_t>();
    size_t bloom_size2 = result["size2"].as<size_t>();
    size_t nb_items = result["items"].as<size_t>();

    bool do_log_perf = result["log"].as<bool>();
    std::string log_params = std::to_string(nb_hash) + "," + std::to_string(bloom_size2) + "," + std::to_string(nb_items);
    if (!do_log_perf) {
        log_params = "";
    }

    spdlog::set_level(spdlog::level::info);

	std::vector<uint64_t> items = get_seq_items(nb_items);
    std::vector<uint64_t> no_items = get_seq_items(NB_NO_ITEMS, nb_items);

	std::cout << "> Creating filter..." << std::endl;
	auto bloom_filter = BloomFilterTimeitDecorator<SyncCacheBloomFilter>(log_params, "bench2_perfs", bloom_size2, nb_hash, NB_THREADS);

    std::cout << "> Inserting many items..." << std::endl;
	bloom_filter.insert_bulk(items);

    std::cout << "> Computing weight..." << std::endl;
    auto weight = bloom_filter.get_weight();
    std::cout << "Weight is " << weight << std::endl;

    std::cout << "> Querying all inserted items in a random order..." << std::endl;
	auto rng = std::default_random_engine{};
	std::shuffle(std::begin(items), std::end(items), rng);
	bloom_filter.contains_bulk(items);

    std::cout << "> Querying non inserted items and checking fpr..." << std::endl;
    auto lookup_result = bloom_filter.contains_bulk(no_items);
	double fpr = (double) std::count(lookup_result.begin(), lookup_result.end(), true) / no_items.size();
    bloom_filter.log_data(fpr, "fpr");
    std::cout << "False positive rate is " << fpr << std::endl;

    // std::cout << "> Getting data..." << std::endl;
    // auto data = bloom_filter.get_data();
    
    // std::cout << "> Creating a new filter..." << std::endl;
    // auto bloom_filter2 = SyncCacheBloomFilter(bloom_size2, nb_hash, NB_THREADS);
    // std::cout << "Weight is " << bloom_filter2.get_weight() << std::endl;

    // std::cout << "> Loading data into the new filter..." << std::endl;
    // bloom_filter2.set_data(data);

    // std::cout << "> Computing weight of new filter after data loading..." << std::endl;
    // std::cout << "Weight is " << bloom_filter2.get_weight() << std::endl;

    // std::cout << "> The end." << std::endl;

}
