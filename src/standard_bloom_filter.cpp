# include "bloom_filter.hpp"

class SyncBasicBloomFilter : public BucketIterativeBloomFilter {
    
    public:

        SyncBasicBloomFilter(size_t size2, size_t nb_hash, size_t nb_threads = 1) : BucketIterativeBloomFilter(size2, nb_hash, nb_threads),
                _hash_functions(nb_hash), _nb_bytes(_get_size() >> 3LU) {

            _bloom_data.resize(_nb_bytes, (uint8_t) 0);

        }

        void insert(const u_int64_t& item) override {
            for (size_t i = 0 ; i < get_nb_hash(); i++) {
                u_int64_t h = _hash_functions(item, i) & _get_size_reduced();
                __sync_fetch_and_or(_bloom_data.data() + (h >> 3LU), _get_bit_mask(h & 7LU));
            }
        }

        bool contains(const u_int64_t& item) override {
            for (size_t i = 0 ; i < get_nb_hash(); i++) {
                u_int64_t h = _hash_functions(item, i) & _get_size_reduced();
				if ((_bloom_data[h >> 3LU] & _get_bit_mask(h & 7LU)) == 0) {
                    return false;
                }
            }
            return true;
        }

        size_t get_weight() override {
            auto thread_results = std::vector<size_t>(_get_nb_threads(), 0);
            #pragma omp parallel for num_threads(_get_nb_threads())
            for (size_t i = 0; i < _bloom_data.size(); i++) {
                thread_results[omp_get_thread_num()] += __builtin_popcount(_bloom_data[i]);
            }
            size_t result = 0;
            for (size_t tid = 0; tid < _get_nb_threads(); tid++) {
                result += thread_results[tid];
            }
            return result;
        }

        const std::vector<uint8_t>& get_data() override {
            return _bloom_data;
        }

        void set_data(const std::vector<uint8_t>& data) override {
            _bloom_data.assign(data.begin(), data.end());
        }
    
    protected:

        uint64_t _hash_for_bucket(const uint64_t& item) {
            return _hash_functions(item, 0) & _get_size_reduced();
        }

        BloomHashFunctions _get_hash_functions() { return _hash_functions; }
        std::vector<uint8_t>& _get_data() { return _bloom_data; }

    private:

        std::vector<uint8_t> _bloom_data;
        BloomHashFunctions _hash_functions;
        size_t _nb_bytes;

};

class BasicBloomFilter : public SyncBasicBloomFilter {

    public:

        BasicBloomFilter(size_t size2, size_t nb_hash, size_t nb_threads = 1) : SyncBasicBloomFilter(size2, nb_hash, nb_threads) {
            
            if (_get_nb_threads() > 1) {
                throw std::invalid_argument(std::string("Error: this class cannot be used with more than 1 thread, use the synchronized equivalent SyncBasicBloomFilter instead"));
            }

        }

        void insert(const u_int64_t& item) override {
            for (size_t i = 0 ; i < get_nb_hash(); i++) {
                u_int64_t h = _get_hash_functions()(item, i) & _get_size_reduced();
                _get_data()[h >> 3LU] |= _get_bit_mask(h & 7LU);
            }
        }

};

class SyncCacheBloomFilter : public BucketIterativeBloomFilter {
    
    public:

        SyncCacheBloomFilter(size_t size2, size_t nb_hash, size_t nb_threads = 1, size_t block_size2 = 6) : BucketIterativeBloomFilter(size2, nb_hash, nb_threads),
                _hash_functions(nb_hash), _nb_bytes(_get_size()), _nb_bytes_extended(_nb_bytes + (((1LU << block_size2)) >> 3LU)), _block_mask((1LU << block_size2) - 1) {
            
            _bloom_data.resize(_nb_bytes_extended, (uint8_t) 0);

        }

        void insert(const u_int64_t& item) override {
            _insert(item, _hash_functions(item, 0) & _get_size_reduced());
        }

        bool contains(const u_int64_t& item) override {
            return _contains(item, _hash_functions(item, 0) & _get_size_reduced());
        }

        size_t get_weight() override {
            auto thread_results = std::vector<size_t>(_get_nb_threads(), 0);
            #pragma omp parallel for num_threads(_get_nb_threads())
            for (size_t i = 0; i < _bloom_data.size(); i++) {
                thread_results[omp_get_thread_num()] += __builtin_popcount(_bloom_data[i]);
            }
            size_t result = 0;
            for (size_t tid = 0; tid < _get_nb_threads(); tid++) {
                result += thread_results[tid];
            }
            return result;
        }

        const std::vector<uint8_t>& get_data() override {
            return _bloom_data;
        }

        void set_data(const std::vector<uint8_t>& data) override {
            _bloom_data.assign(data.begin(), data.end());
        }
    
    protected:

        uint64_t _hash_for_bucket(const uint64_t& item) {
            return _hash_functions(item, 0) & _get_size_reduced();
        }

        virtual void _insert(const u_int64_t& item, const u_int64_t& h0) {
            __sync_fetch_and_or(_bloom_data.data() + (h0 >> 3LU), _get_bit_mask(h0 & 7LU));
            for (size_t i = 1 ; i < get_nb_hash(); i++) {
                u_int64_t h = h0  + (_hash_functions.simplehash16_64(item, i) & _block_mask);
                __sync_fetch_and_or(_bloom_data.data() + (h >> 3LU), _get_bit_mask(h & 7LU));
            }
        }

        bool _contains(const u_int64_t& item, const u_int64_t& h0) {
            if ((_bloom_data[h0 >> 3LU] & _get_bit_mask(h0 & 7LU)) == 0) {
                return false;
            }
            for (size_t i = 1 ; i < get_nb_hash(); i++) {
                u_int64_t h = h0 + (_hash_functions.simplehash16_64(item, i) & _block_mask);
				if ((_bloom_data[h >> 3LU] & _get_bit_mask(h & 7LU)) == 0) {
                    return false;
                }
            }
            return true;
        }

        BloomHashFunctions _get_hash_functions() { return _hash_functions; }
        std::vector<uint8_t>& _get_data() { return _bloom_data; }
        size_t _get_block_mask() { return _block_mask; }

    private:

        std::vector<uint8_t> _bloom_data;
        BloomHashFunctions _hash_functions;
        size_t _nb_bytes;
        size_t _nb_bytes_extended;
        size_t _block_mask;

};

class CacheBloomFilter : public SyncCacheBloomFilter {

    public:

        CacheBloomFilter(size_t size2, size_t nb_hash, size_t nb_threads = 1, size_t block_size2 = 6) : SyncCacheBloomFilter(size2, nb_hash, nb_threads, block_size2) {
            
            if (_get_nb_threads() > 1) {
                throw std::invalid_argument(std::string("Error: this implementation cannot be used with more than 1 thread, use the synchronized equivalent SyncCacheBloomFilter instead"));
            }

        }

    protected:

        void _insert(const u_int64_t& item, const u_int64_t& h0) override {
            _get_data()[h0 >> 3LU] |= _get_bit_mask(h0 & 7LU);
            for (size_t i = 1 ; i < get_nb_hash(); i++) {
                u_int64_t h = h0 + (_get_hash_functions().simplehash16_64(item, i) & _get_block_mask());
                _get_data()[h >> 3LU] |= _get_bit_mask(h & 7LU);
            }
        }

};