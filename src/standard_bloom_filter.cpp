# include "bloom_filter.hpp"

class SyncBasicBloomFilter : public IterativeBloomFilter {
    
    public:

        SyncBasicBloomFilter(size_t size2, size_t nb_hash, size_t nb_threads = 1) : IterativeBloomFilter(size2, nb_hash, nb_threads),
                _hash_functions(nb_hash), _nb_bits(_get_size() >> 3LU) {

            _bloom_data.resize(_nb_bits, (uint8_t) 0);
            _nb_bits--;

        }

        void insert(const u_int64_t& item) override {
            for (size_t i = 0 ; i < get_nb_hash(); i++) {
                u_int64_t h = _hash_functions(item, i) & _nb_bits;
                __sync_fetch_and_or(_bloom_data.data() + (h >> 3LU), _get_bit_mask(h & 7LU));
            }
        }

        bool contains(const u_int64_t& item) override {
            for (size_t i = 0 ; i < get_nb_hash(); i++) {
                u_int64_t h = _hash_functions(item, i) & _nb_bits;
				if ((_bloom_data[h >> 3LU] & _get_bit_mask(h & 7LU)) == 0) {
                    return false;
                }
            }
            return true;
        }

        size_t get_weight() override {
            size_t result = 0;
            for (auto data : _bloom_data) {
                result += __builtin_popcount(data);
            }
            return result;
        }

        const std::vector<uint8_t>& get_data() override {
            return _bloom_data;
        }
    
    protected:

        BloomHashFunctions _get_hash_functions() { return _hash_functions; }
        std::vector<uint8_t>& _get_data() { return _bloom_data; }
        size_t _get_nb_bits() { return _nb_bits; }

    private:

        std::vector<uint8_t> _bloom_data;
        BloomHashFunctions _hash_functions;
        size_t _nb_bits;

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
                u_int64_t h = _get_hash_functions()(item, i) & _get_nb_bits();
                _get_data()[h >> 3LU] |= _get_bit_mask(h & 7LU);
            }
        }

};

class SyncCacheBloomFilter : public IterativeBloomFilter {
    
    public:

        SyncCacheBloomFilter(size_t size2, size_t nb_hash, size_t nb_threads = 1, size_t block_size2 = 6) : IterativeBloomFilter(size2, nb_hash, nb_threads),
                _hash_functions(nb_hash), _nb_bits(_get_size()), _nb_bits_extended(_nb_bits + (((1LU << block_size2)) >> 3LU)), _block_mask((1LU << block_size2) - 1) {
            
            _bloom_data.resize(_nb_bits_extended, (uint8_t) 0);
            _nb_bits--;

        }

        void insert(const u_int64_t& item) override {
            _insert(item, _hash_functions(item, 0) & _nb_bits);
        }

        bool contains(const u_int64_t& item) override {
            return _contains(item, _hash_functions(item, 0) & _nb_bits);
        }

        size_t get_weight() override {
            size_t result = 0;
            for (auto data : _bloom_data) {
                result += __builtin_popcount(data);
            }
            return result;
        }

        const std::vector<uint8_t>& get_data() override {
            return _bloom_data;
        }
    
    protected:

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
        size_t _get_nb_bits() { return _nb_bits; }
        size_t _get_block_mask() { return _block_mask; }

    private:

        std::vector<uint8_t> _bloom_data;
        BloomHashFunctions _hash_functions;
        size_t _nb_bits;
        size_t _nb_bits_extended;
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