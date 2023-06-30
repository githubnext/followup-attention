/*****************************************************************************/

/*
This algorithms concerns prime numbers: number which are divisible only by

themselves and by 1.
*/

#include <stdlib.h>
#include <iostream>
#include <vector>
using namespace std;

int countPrimes(int n)
{
    vector<bool> isPrimer(n, true);

    for (int i = 2; i * i < n; i++)
    {
        if (isPrimer[i])
        {
            for (int j = i * i; j < n; j += i)
            {
                isPrimer[j] = false;
            }
        }
    }

    int cnt = 0;
    for (int i = 2; i < n; i++)
    {
        if (isPrimer[i])
        {
            cnt++;
        }
    }
    return cnt;
}

int main(int argc, char **argv)
{
    int n = 100;
    if (argc > 1)
    {
        n = atoi(argv[1]);
    }

    cout << endl
         << n << " : " << countPrimes(n) << endl;

    return 0;
}

/*
Question: What is the complexity of this algorithm?

Answer:
*/