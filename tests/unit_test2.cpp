#include <catch2/catch_test_macros.hpp>

#include <stdint.h>
#include <cstdio>
#include <iostream>
#include <random>
#include <algorithm>

#include "tests_utils.hpp"
#include "bloom_filter.hpp"
#include "standard_bloom_filter.cpp"
#include "pim_bloom_filter.cpp"

#define NB_THREADS 8

#define NB_ITEMS 10000
#define NB_NO_ITEMS 1000

TEST_CASE("Testing Bloom filters with sync basic implementation") {

	std::vector<uint64_t> items = get_seq_items(NB_ITEMS);
	std::vector<uint64_t> no_items = get_seq_items(NB_NO_ITEMS, NB_ITEMS);

	INFO("Creating filter...");

	size_t bloom_size2, nb_hash;

	SECTION("") {
		bloom_size2 = 24;
		SECTION("") { nb_hash = 1; }
		SECTION("") { nb_hash = 8; }
	}
	SECTION("") {
		bloom_size2 = 16;
		nb_hash = 2;
	}

	CAPTURE(bloom_size2);
	CAPTURE(nb_hash);

	auto bloom_filter = SyncBasicBloomFilter(bloom_size2, nb_hash, NB_THREADS);

	INFO("Checking weight after initialization...");
	size_t weight = bloom_filter.get_weight();
	CHECK(weight == 0);

	INFO("Inserting 1 item and checking weight...");
	bloom_filter.insert(1);
	weight = bloom_filter.get_weight();
	CHECK(weight > 0);
	CHECK(weight <= nb_hash);
	CHECK(weight <= (1 << bloom_size2));

    INFO("Checking single lookup...");
    CHECK(bloom_filter.contains(1));

	// Insertions should be deterministic
	INFO("Inserting the same item again and checking weight did not change...");
	bloom_filter.insert(1);
	CHECK(weight == bloom_filter.get_weight());

	INFO("Inserting many items...");
	bloom_filter.insert_bulk(items);

	INFO("Checking there is no false negatives...");
	auto rng = std::default_random_engine{};
	std::shuffle(std::begin(items), std::end(items), rng);
	auto result = bloom_filter.contains_bulk(items);
	CHECK(static_cast<size_t>(std::count(result.begin(), result.end(), true)) == items.size());

	INFO("Checking rate of false positives...");
	result = bloom_filter.contains_bulk(no_items);
	double fpr = (double) std::count(result.begin(), result.end(), true) / no_items.size();
	CHECK(fpr <= 1.0);
	if (items.size() <= (size_t) (1 << bloom_size2)) {
		if (fpr > 0.1) {
			CAPTURE(fpr);
			WARN(std::string("False positive rate is significantly high: ") + std::to_string(fpr));
		}
	}

	INFO("Checking lookups...");
	std::vector<uint64_t> other_items = {items[0], no_items[0]};
	result = bloom_filter.contains_bulk(other_items);
	CHECK(result[0] == true);

	other_items = {no_items[0], items[0]};
	result = bloom_filter.contains_bulk(other_items);
	CHECK(result[1] == true);

}