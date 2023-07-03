TARGET_HOST = bloom_filters_host
TARGETS_DPU = $(patsubst %.c,%,$(wildcard bloom_filters_dpu*.c))
TARGETS_DPU_H = bloom_filters_dpu.h
COMMON_H = bloom_filters_common.h

make: $(TARGETS_DPU) $(TARGET_HOST)

$(TARGET_HOST): %: %.cpp $(COMMON_H)
	g++ --std=c++11 $< -o $@ `dpu-pkg-config --cflags --libs dpu` -g
	
$(TARGETS_DPU): %: %.c bloom_filters_common.h $(COMMON_H) $(TARGETS_DPU_H)
	dpu-upmem-dpurte-clang $< -DNR_TASKLETS=16 -DSTACK_SIZE_DEFAULT=3072 -O2 -o $@

clean:
	rm -f $(TARGET_HOST) $(TARGETS_DPU)