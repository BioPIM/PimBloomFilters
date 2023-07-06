#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "thirdparty/doctest/doctest.h"

#include <stdint.h>
#include <stdio.h>
#include <iostream>
#include <random>
#include <algorithm>

#include "test.hpp"
#include "bloom_filters.cpp"

#define NB_ITEMS 10000
#define NB_NO_ITEMS 1000

TEST_CASE("Testing Bloom filters with simulator") {

	std::vector<uint64_t> items = get_seq_items(NB_ITEMS);
	std::vector<uint64_t> no_items = get_seq_items(NB_NO_ITEMS, NB_ITEMS);

	INFO("Creating filter...");
	PimBloomFilter *bloom_filter;

	int nb_dpu, bloom_size2, nb_hash;
	const char* dpu_profile = DpuProfile::SIMULATOR;

	SUBCASE("") {
		nb_dpu = 1;
		bloom_size2 = 20;
		SUBCASE("") { nb_hash = 1; }
		SUBCASE("") { nb_hash = 8; }
	}
	SUBCASE("") {
		nb_dpu = 4;
		bloom_size2 = 24;
		SUBCASE("") { nb_hash = 1; }
		SUBCASE("") { nb_hash = 8; }
	}
	SUBCASE("") {
		nb_dpu = 1;
		bloom_size2 = 4;
		nb_hash = 4;
	}

	CAPTURE(nb_dpu);
	CAPTURE(bloom_size2);
	CAPTURE(nb_hash);

	bloom_filter = new PimBloomFilter(nb_dpu, bloom_size2, nb_hash, PimBloomFilter::BASIC_CACHE_ITEMS, dpu_profile);

	INFO("Checking weight after initialization...");
	uint32_t weight = bloom_filter->get_weight();
	CHECK(weight == 0);

	INFO("Inserting 1 item and checking weight...");
	bloom_filter->insert(1);
	weight = bloom_filter->get_weight();
	CHECK(weight > 0);
	CHECK(weight <= nb_hash);
	CHECK(weight <= (1 << bloom_size2));

	// Insertions should be deterministic
	INFO("Inserting the same item again and checking weight did not change...");
	bloom_filter->insert(1);
	CHECK(weight == bloom_filter->get_weight());

	INFO("Inserting many items...");
	bloom_filter->insert(items);

	INFO("Checking there is no false negatives...");
	auto rng = std::default_random_engine{};
	std::shuffle(std::begin(items), std::end(items), rng);
	uint32_t nb_positive_lookups = bloom_filter->contains(items);
	CHECK(nb_positive_lookups == items.size());

	INFO("Checking rate of false positives...");
	nb_positive_lookups = bloom_filter->contains(no_items);
	double fpr = (double) nb_positive_lookups / no_items.size();
	CHECK(fpr <= 1.0);
	if (items.size() <= (1 << bloom_size2)) {
		WARN(fpr <= 0.1);
	}

	delete bloom_filter;

}
