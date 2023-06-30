/*# ***********************************************************************************************

# In the Consumer-Producer problem, two threads types, called producers and

# consumers, share the same memory buffer that is of fixed-size.

# The producers add data to the buffer, whereas the consumers remove data.

*/

#include <bits/stdc++.h>
#include <thread>
using namespace std;

class Queue {
    int maxSize;
    vector<int> q;

    public:
    Queue(int maxSize) {
        this->maxSize = maxSize;
    }
    void put(int item) {
        if (q.size() == maxSize) {
            cout << "Queue is full" << endl;
            throw runtime_error("Queue is full");
        }
        q.push_back(item);
    }
    int get() {
        if (q.size() == 0) {
            cout << "Queue is empty" << endl;
            throw runtime_error("Queue is empty");
        }
        int item = q[0];
        q.erase(q.begin());
        return item;
    }
    bool isFull() {
        return q.size() == maxSize;
    }
    bool isEmpty() {
        return q.size() == 0;
    }
    string toString() {
        stringstream ss;
        for (int i = 0; i < q.size(); i++) {
            ss << q[i] << " ";
        }
        return ss.str();
    }
};

int BUF_SIZE = 10;
Queue myQueue(BUF_SIZE);

void producerRun() {
    while (true) {
        if (!myQueue.isFull()) {
            int item = rand() % 10;
            myQueue.put(item);
            std::thread::id currentThreadId = std::this_thread::get_id();
            cout << "Produced (" << currentThreadId  << "): " << item <<
                    " - " << myQueue.toString() << endl;
            this_thread::sleep_for(chrono::microseconds(1));
        }
    }
}


void consumerRun() {
    while (true) {
        if (!myQueue.isEmpty()) {
            int item = myQueue.get();
            std::thread::id currentThreadId = std::this_thread::get_id();
            cout << "Consumed (" << currentThreadId  << "): " << item <<
                    " - " << myQueue.toString() << endl;
            this_thread::sleep_for(chrono::microseconds(1));
        }
    }
}



int main(int argc, char const *argv[])
{
  std::cout << "Hello Docker container!" << std::endl;
    std::thread producerThreads[10];
    for (int i = 0; i < 10; i++) {
        producerThreads[i] = std::thread(producerRun);
    }

    thread consumer(consumerRun);

    for (int i = 0; i < 10; i++) {
        producerThreads[i].join();
    }
    consumer.join();

  return 0;
}

/*

# Questions: is it possible that consumer and producers threads

# end up in a deadlock state, namely they both wait for each other to finish,

# but none of them is doing anything?

# Answer:
*/