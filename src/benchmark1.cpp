#include "thirdparty/cxxopts/cxxopts.hpp"

#include <stdint.h>
#include <stdio.h>
#include <iostream>
#include <random>
#include <algorithm>

#include "run_utils.hpp"
#include "bloom_filters.cpp"

#define BLOOM_SIZE (1 << BLOOM_SIZE2)

int main(int argc, char** argv) {

    cxxopts::Options options("main_test2", "Run and time PIM Bloom filters micro-benchmarks");

    options.add_options()
        ("k,hash", "Number of hash functions", cxxopts::value<int>()->default_value("8"))
        ("r,rank", "Number of DPUs ranks", cxxopts::value<int>()->default_value("1"))
        ("m,size2", "Size2 of the filter", cxxopts::value<int>()->default_value("20"))
        ("n,items", "Number of items", cxxopts::value<int>()->default_value("10000"))
        ("s,simulator", "Use the simulator", cxxopts::value<bool>()->default_value("false"))
        ("t,trace", "Enable trace watchers", cxxopts::value<bool>()->default_value("false"))
        ("h,help", "Print usage")
    ;

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    int nb_ranks = result["rank"].as<int>();
    int nb_hash = result["hash"].as<int>();
    int bloom_size2 = result["size2"].as<int>();
    int nb_items = result["items"].as<int>();
    const char* dpu_profile = result["simulator"].as<bool>() ? DpuProfile::SIMULATOR : DpuProfile::HARDWARE;

	std::vector<uint64_t> items = get_seq_items(nb_items);

	std::cout << "> Creating filter..." << std::endl;
	PimBloomFilter *bloom_filter;
    TIMEIT(bloom_filter = new PimBloomFilter(nb_ranks, bloom_size2, nb_hash, PimBloomFilter::BASIC_CACHE_ITEMS, dpu_profile, result["trace"].as<bool>()));

    std::cout << "> Inserting many items..." << std::endl;
	TIMEIT(bloom_filter->insert(items));

    std::cout << "> Computing weight..." << std::endl;
    uint32_t weight;
    TIMEIT(weight = bloom_filter->get_weight());
    std::cout << "Weight is " << weight << std::endl;

    // std::cout << "> Querying all inserted items in a random order..." << std::endl;
	// auto rng = std::default_random_engine{};
	// std::shuffle(std::begin(items), std::end(items), rng);
	// TIMEIT(bloom_filter->contains(items));

    std::cout << "> Cleaning filter..." << std::endl;
	delete bloom_filter;

    std::cout << "> The end." << std::endl;

}
