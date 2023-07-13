TARGET_TESTS = $(patsubst %.cpp,%,$(wildcard main_test*.cpp))
TARGETS_DPU = $(patsubst %.c,%,$(wildcard bloom_filters_dpu*.c))
TARGETS_DPU_H = bloom_filters_dpu.h
COMMON_H = bloom_filters_common.h
BLOOM_CPP = bloom_filters.cpp pim_rankset.cpp
TEST_H = test.hpp

make: $(TARGETS_DPU) $(TARGET_TESTS)

$(TARGET_TESTS): %: %.cpp $(COMMON_H) $(TEST_H) $(BLOOM_CPP)
	g++ --std=c++11 $< -o $@ `dpu-pkg-config --cflags --libs dpu` -fopenmp -g
	
$(TARGETS_DPU): %: %.c bloom_filters_common.h $(COMMON_H) $(TARGETS_DPU_H)
	dpu-upmem-dpurte-clang $< -DNR_TASKLETS=16 -DSTACK_SIZE_DEFAULT=3072 -O2 -o $@

clean:
	rm -f $(TARGET_TESTS) $(TARGETS_DPU)