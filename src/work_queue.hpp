#include <mutex>
#include <queue>
#include <chrono>
#include <thread>
#include <utility>
#include <functional>

template<class T>
class WorkConsumer {
    public:
        virtual void run(T data) = 0;
};

template<class T>
class WorkQueue {

    public:

        WorkQueue(size_t nb_producers) : _nb_producers(nb_producers) {}
        ~WorkQueue() = default;

        void produce(T& data) {
            _queue_mutex.lock();
            _queue.push(std::move(data));
            _queue_mutex.unlock();
        }

        void done_producing() {
            _done_producing_mutex.lock();
            _nb_producers_done += 1;
            _done_producing_mutex.unlock();
        }

        void consume(WorkConsumer consumer_) {
            while(true) {
                _queue_mutex.lock();
                if (_queue.empty()) {
                    _queue_mutex.unlock();
                    _done_producing_mutex.lock();
                    bool no_more_work = (_nb_producers_done >= _nb_producers);
                    _done_producing_mutex.unlock();
                    if (no_more_work) {
                        break;
                    }
                } else {
                    auto data = _queue.front();
                    _queue.pop();
                    _queue_mutex.unlock();
                    consumer.run(data);
                }
                std::this_thread::sleep_for(_CONSUMER_WAIT);
            }
        }

    private:

        size_t _nb_producers;
        size_t _nb_producers_done = 0;
		std::mutex _done_producing_mutex;
        std::queue<T> _queue;
        std::mutex _queue_mutex;

        const auto _CONSUMER_WAIT = std::chrono::milliseconds(10);

};