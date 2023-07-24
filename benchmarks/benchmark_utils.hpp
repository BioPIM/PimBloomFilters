#ifndef FD206EEF_A5B2_45D1_9607_FA5C11F4B995
#define FD206EEF_A5B2_45D1_9607_FA5C11F4B995

#include <vector>
#include <omp.h>
#include <cstdint>
#include <cstdio>

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


#endif /* FD206EEF_A5B2_45D1_9607_FA5C11F4B995 */
