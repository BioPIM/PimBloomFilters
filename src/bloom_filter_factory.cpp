#ifndef F4ED384A_FF47_4581_9604_CE4F2D68E017
#define F4ED384A_FF47_4581_9604_CE4F2D68E017

#include "standard_bloom_filter.cpp"
#include "pim_bloom_filter.cpp"

enum BloomFilterType {
    SYNC_BASIC,
    BASIC,
    SYNC_CACHE,
    CACHE,
    PIM,
};

class BloomFilterFactory {

    public:

        static std::unique_ptr<IBloomFilter> create_filter(BloomFilterType type, size_t size2, size_t nb_hash, size_t nb_threads = 1) {
            
            switch(type) {

                case BloomFilterType::SYNC_BASIC:
                    return std::make_unique<SyncBasicBloomFilter>(size2, nb_hash, nb_threads);
                
                case BloomFilterType::BASIC:
                    if (nb_threads > 1) {
                        spdlog::warn("BasicBloomFilter has no sync, modified arg to be only 1 thread");
                    }
                    return std::make_unique<BasicBloomFilter>(size2, nb_hash, 1);
                
                case BloomFilterType::SYNC_CACHE:
                    return std::make_unique<SyncCacheBloomFilter>(size2, nb_hash, nb_threads);

                case BloomFilterType::CACHE:
                    if (nb_threads > 1) {
                        spdlog::warn("CacheBloomFilter has no sync, modified arg to be only 1 thread");
                    }
                    return std::make_unique<CacheBloomFilter>(size2, nb_hash, 1);
                
                case BloomFilterType::PIM:
                    return std::make_unique<PimBloomFilter<HashPimItemDispatcher>>(size2, nb_hash, nb_threads);

                default:
                    throw std::invalid_argument(std::string("Unknown filter type"));
            }
        }

};

#endif /* F4ED384A_FF47_4581_9604_CE4F2D68E017 */
