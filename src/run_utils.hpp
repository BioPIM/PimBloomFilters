#include <vector>
#include <omp.h>
#include <stdint.h>

#define TIMEIT(f) \
    do { \
        double start = omp_get_wtime(); \
        f; \
        double stop = omp_get_wtime(); \
        std::cout << "Took " << stop - start << " seconds" << std::endl; \
    } while (0)

std::vector<uint64_t> get_seq_items(const size_t nb, const uint64_t start_offset = 0) {
    std::vector<uint64_t> items(nb);
	for (size_t i = 0; i < nb; i++) {
		items[i] = i + start_offset;
	}
    return items;
}