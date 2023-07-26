#include "cxxopts/cxxopts.hpp"
#include "spdlog/spdlog.h"

#include <stdint.h>
#include <stdio.h>
#include <iostream>
#include <random>
#include <algorithm>

#include "benchmark_utils.hpp"
#include "standard_bloom_filter.cpp"

constexpr size_t NB_THREADS = 8; 

int main(int argc, char** argv) {

    cxxopts::Options options("benchmark2", "Run and time standard Bloom filters micro-benchmarks");

    options.add_options()
        ("k,hash", "Number of hash functions", cxxopts::value<size_t>()->default_value("8"))
        ("m,size2", "Size2 of the filter", cxxopts::value<size_t>()->default_value("20"))
        ("n,items", "Number of items", cxxopts::value<size_t>()->default_value("10000"))
        ("h,help", "Print usage")
    ;

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    size_t nb_hash = result["hash"].as<size_t>();
    size_t bloom_size2 = result["size2"].as<size_t>();
    size_t nb_items = result["items"].as<size_t>();

	std::vector<uint64_t> items = get_seq_items(nb_items);

    spdlog::set_level(spdlog::level::info);

	std::cout << "> Creating filter..." << std::endl;
	std::unique_ptr<IBloomFilter> bloom_filter;
    TIMEIT(bloom_filter = std::make_unique<SyncCacheBloomFilter>(bloom_size2, nb_hash, NB_THREADS));

    std::cout << "> Inserting many items..." << std::endl;
	TIMEIT(bloom_filter->insert_bulk(items));

    // std::cout << "> Computing weight..." << std::endl;
    // size_t weight;
    // TIMEIT(weight = bloom_filter->get_weight());
    // std::cout << "Weight is " << weight << std::endl;

    // std::cout << "> Querying all inserted items in a random order..." << std::endl;
	// auto rng = std::default_random_engine{};
	// std::shuffle(std::begin(items), std::end(items), rng);
	// TIMEIT(bloom_filter->contains_bulk(items));

    std::cout << "> The end." << std::endl;

}
