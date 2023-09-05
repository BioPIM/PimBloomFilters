#ifndef B11E1701_AF58_4807_9F81_2E6FFDD82EBD
#define B11E1701_AF58_4807_9F81_2E6FFDD82EBD

#include <vector>
#include <cstdint>
#include <cstdio>


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


#endif /* B11E1701_AF58_4807_9F81_2E6FFDD82EBD */
