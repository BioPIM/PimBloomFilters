#include "cxxopts/cxxopts.hpp"
#include "spdlog/spdlog.h"

#include <stdint.h>
#include <stdio.h>
#include <iostream>
#include <random>
#include <algorithm>

#include "benchmark_utils.hpp"
#include "pim_bloom_filter.cpp"

constexpr size_t NB_THREADS = 8; 

int main(int argc, char** argv) {

    cxxopts::Options options("benchmark1", "Run and time PIM Bloom filters micro-benchmarks");

    options.add_options()
        ("k,hash", "Number of hash functions", cxxopts::value<size_t>()->default_value("8"))
        ("r,rank", "Number of DPUs ranks", cxxopts::value<size_t>()->default_value("1"))
        ("m,size2", "Size2 of the filter", cxxopts::value<size_t>()->default_value("20"))
        ("n,items", "Number of items", cxxopts::value<size_t>()->default_value("10000"))
        ("s,simulator", "Use the simulator", cxxopts::value<bool>()->default_value("false"))
        ("h,help", "Print usage")
    ;

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    size_t nb_ranks = result["rank"].as<size_t>();
    size_t nb_hash = result["hash"].as<size_t>();
    size_t bloom_size2 = result["size2"].as<size_t>();
    size_t nb_items = result["items"].as<size_t>();
    DpuProfile dpu_profile = result["simulator"].as<bool>() ? DpuProfile::SIMULATOR : DpuProfile::HARDWARE;

	std::vector<uint64_t> items = get_seq_items(nb_items);

    spdlog::set_level(spdlog::level::info);

	std::cout << "> Creating filter..." << std::endl;
	std::unique_ptr<IBloomFilter> bloom_filter;
    TIMEIT(bloom_filter = std::make_unique<PimBloomFilter<HashPimItemDispatcher>>(bloom_size2, nb_hash, NB_THREADS, nb_ranks, dpu_profile));

    std::cout << "> Inserting many items..." << std::endl;
	TIMEIT(bloom_filter->insert_bulk(items));

    std::cout << "> Computing weight..." << std::endl;
    size_t weight;
    TIMEIT(weight = bloom_filter->get_weight());
    std::cout << "Weight is " << weight << std::endl;

    std::cout << "> Querying all inserted items in a random order..." << std::endl;
	auto rng = std::default_random_engine{};
	std::shuffle(std::begin(items), std::end(items), rng);
	TIMEIT(bloom_filter->contains_bulk(items));

    // std::cout << "> Getting data..." << std::endl;
    // std::vector<uint8_t> data;
    // TIMEIT(data = bloom_filter->get_data());
    
    // std::cout << "> Creating a new filter..." << std::endl;
    // std::unique_ptr<IBloomFilter> bloom_filter2 = std::make_unique<PimBloomFilter<HashPimItemDispatcher>>(bloom_size2, nb_hash, NB_THREADS, nb_ranks, dpu_profile);
    // std::cout << "Weight is " << bloom_filter2->get_weight() << std::endl;

    // std::cout << "> Loading data into the new filter..." << std::endl;
    // TIMEIT(bloom_filter2->set_data(data));

    // std::cout << "> Computing weight of new filter after data loading..." << std::endl;
    // std::cout << "Weight is " << bloom_filter2->get_weight() << std::endl;

    std::cout << "> The end." << std::endl;

}
