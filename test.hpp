#include <vector>
#include <omp.h>

#define TIMEIT(f) \
    do { \
        double start = omp_get_wtime(); \
        f; \
        double stop = omp_get_wtime(); \
        std::cout << "Took " << stop - start << " seconds" << std::endl; \
    } while (0)

class DpuProfile {
public:
    const static char* HARDWARE;
    const static char* SIMULATOR;
};

const char* DpuProfile::HARDWARE =  "backend=hw";
const char* DpuProfile::SIMULATOR =  "backend=simulator";

std::vector<uint64_t> get_seq_items(const int nb, const int start_offset = 0) {
    std::vector<uint64_t> items(nb);
	for (int i = 0; i < nb; i++) {
		items[i] = i + start_offset;
	}
    return items;
}