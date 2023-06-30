# ***********************************************************************************************

# In the Consumer-Producer problem, two threads types, called producers and

# consumers, share the same memory buffer that is of fixed-size.

# The producers add data to the buffer, whereas the consumers removes data.


import threading
import time
import random
import logging


logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s',)


class Queue(object):

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.q = []

    def put(self, item):
        if len(self.q) == self.max_size:
            raise Exception('Queue is full')
        self.q.append(item)

    def get(self):
        if len(self.q) == 0:
            raise Exception('Queue is empty')
        return self.q.pop(0)

    def is_full(self):
        return len(self.q) == self.max_size

    def is_empty(self):
        return len(self.q) == 0

    def __repr__(self):
        return str(self.q)


BUF_SIZE = 10
q = Queue(max_size=BUF_SIZE)


class ProducerThread(threading.Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs=None, verbose=None):
        super(ProducerThread, self).__init__()
        self.target = target
        self.name = name

    def run(self):
        global q
        while True:
            if not q.is_full():
                item = random.randint(1, 10)
                q.put(item)
                logging.debug('Putting ' + str(item)
                              + ' : items in queue ' + str(q))
                time.sleep(random.random() / 1000)
        return


class ConsumerThread(threading.Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs=None, verbose=None):
        super(ConsumerThread, self).__init__()
        self.target = target
        self.name = name
        return

    def run(self):
        global q
        while True:
            if not q.is_empty():
                item = q.get()
                logging.debug('Getting ' + str(item)
                              + ' : items in queue ' + str(q))
                time.sleep(random.random() / 1000)
        return


if __name__ == '__main__':

    for i in range(10):
        px = ProducerThread(name='producer-' + str(i))
        px.start()
    time.sleep(2)

    c = ConsumerThread(name='consumer')
    c.start()
    time.sleep(2)


# Questions: is it possible that consumer and producers threads

# end up in a deadlock state, namely they both wait for each other to finish,

# but none of them is doing anything?

# Answer:
