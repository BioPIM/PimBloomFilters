# include "bloom_filter.hpp"

class BasicBloomFilter : public IterativeBloomFilter {
    
    public:

        BasicBloomFilter(size_t size2, size_t nb_hash, size_t nb_threads = 1) : IterativeBloomFilter(size2, nb_hash, nb_threads),
                _hash_functions(nb_hash), _nb_bits(_get_size() >> 3LU) {
            _bloom_data.resize(_nb_bits, (uint8_t) 0);
            _nb_bits--;
        }

        void insert(const u_int64_t& item) override {
            for (size_t i = 0 ; i < get_nb_hash(); i++) {
                u_int64_t h = _hash_functions(item, i) & _nb_bits;
                _bloom_data[h >> 3LU] |= _get_bit_mask(h & 7LU);
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

class SyncBasicBloomFilter : public BasicBloomFilter {

    public:

        SyncBasicBloomFilter(size_t size2, size_t nb_hash, size_t nb_threads = 1) : BasicBloomFilter(size2, nb_hash, nb_threads) {};

        void insert(const u_int64_t& item) override {
            for (size_t i = 0 ; i < get_nb_hash(); i++) {
                u_int64_t h = _get_hash_functions()(item, i) & _get_nb_bits();
                __sync_fetch_and_or(_get_data().data() + (h >> 3LU), _get_bit_mask(h & 7LU));
            }
        }

};