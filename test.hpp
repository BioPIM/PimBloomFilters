#include <vector>

#define TIMEIT(f) \
    do { \
        clock_t start = clock(); \
        f; \
        clock_t stop = clock(); \
        std::cout << "Took " << (double) (stop - start) / CLOCKS_PER_SEC << " seconds" << std::endl; \
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