/*# ***********************************************************************************************

# In the Consumer-Producer problem, two threads types, called producers and

# consumers, share the same memory buffer that is of fixed-size.

# The producers add data to the buffer, whereas the consumers remove data.

*/

using System;
using System.Threading;


public class ProducerConsumer {

    public class Queue {
        private List<int> queue;
        private int maxSize;

        public Queue(int maxSize) {
            this.maxSize = maxSize;
            queue = new List<int>();
        }

        public void put(int item) {
            if (queue.Count == maxSize) {
                throw new Exception("Queue is full");
            }
            queue.Add(item);
        }

        public int get() {
            if (queue.Count == 0) {
                throw new Exception("Queue is empty");
            }
            int item = queue[0];
            queue.RemoveAt(0);
            return item;
        }

        public bool isFull() {
            return queue.Count == maxSize;
        }

        public bool isEmpty() {
            return queue.Count == 0;
        }

        public string toString() {
            string str = "";
            foreach (int item in queue) {
                str += item + ", ";
            }
            return str;
        }

    }

    public static int BUF_SIZE = 10;
    public static Queue queue = new Queue(BUF_SIZE);

    public static void producerRun() {
        Random r = new Random();
        while (true) {
            if (!queue.isFull()) {
                int item = r.Next(0, 10);
                Console.WriteLine("Produced: " + item + " - " + queue.toString());
                queue.put(item);
                Thread.Sleep(1);
            }
        }
    }

    public static void consumerRun() {
        while (true) {
            if (!queue.isEmpty()) {
                int item = queue.get();
                Console.WriteLine("Consumed: " + item + " - " + queue.toString());
                Thread.Sleep(1);
            }
        }
    }

    public static void Main() {

        List<Thread> threads = new List<Thread>();
        // start ten producer threads
        for (int i = 0; i < 10; i++) {
            Thread t = new Thread(producerRun);
            t.Start();
            threads.Add(t);
        }

        // start consumer thread
        Thread consumer1 = new Thread(new ThreadStart(consumerRun));
        consumer1.Start();

        // Wait for threads to finish.
        foreach (Thread t in threads) {
            t.Join();
        }
        consumer1.Join();
    }
}

/*# Questions: is there any line of code in the consumer or producer code which

# will never be executed? If yes, report it below.

# Answer:
*/