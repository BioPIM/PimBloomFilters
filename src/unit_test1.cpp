#include <catch2/catch_test_macros.hpp>

#include <stdint.h>
#include <stdio.h>
#include <iostream>
#include <random>
#include <algorithm>

#include "run_utils.hpp"
#include "bloom_filters.cpp"

#define NB_ITEMS 10000
#define NB_NO_ITEMS 1000

TEST_CASE("Testing Bloom filters with simulator") {

	std::vector<uint64_t> items = get_seq_items(NB_ITEMS);
	std::vector<uint64_t> no_items = get_seq_items(NB_NO_ITEMS, NB_ITEMS);

	INFO("Creating filter...");
	PimBloomFilter *bloom_filter;

	int nb_ranks, bloom_size2, nb_hash;
	const char* dpu_profile = DpuProfile::SIMULATOR;

	SECTION("") {
		nb_ranks = 1;
		bloom_size2 = 20;
		SECTION("") { nb_hash = 1; }
		SECTION("") { nb_hash = 8; }
	}
	SECTION("") {
		nb_ranks = 2;
		bloom_size2 = 24;
		SECTION("") { nb_hash = 1; }
		SECTION("") { nb_hash = 8; }
	}
	SECTION("") {
		nb_ranks = 2;
		bloom_size2 = 8;
		nb_hash = 4;
	}

	CAPTURE(nb_ranks);
	CAPTURE(bloom_size2);
	CAPTURE(nb_hash);

	bloom_filter = new PimBloomFilter(nb_ranks, bloom_size2, nb_hash, dpu_profile);

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
		if (fpr > 0.1) {
			CAPTURE(fpr);
			WARN("False positive rate is significantly high");
		}
	}

	delete bloom_filter;

}
