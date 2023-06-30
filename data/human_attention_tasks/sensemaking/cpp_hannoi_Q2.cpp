/*************************************************************************************************/


/*
# In the Tower of Hanoi problem you have to move all the disks

# from the left hand post to the right hand post.

# You can only move the disks one at a time and you can never

# place a bigger disk on a smaller disk.
 */

#include <bits/stdc++.h>
using namespace std;

void towerOfHanoi(int n, char from_rod,
					char to_rod, char aux_rod)
{
	if (n == 0)
	{
		return;
	}
	towerOfHanoi(n - 1, from_rod, aux_rod, to_rod);
	cout << "Move disk " << n << " from rod " << from_rod <<
								" to rod " << to_rod << endl;
	towerOfHanoi(n - 1, aux_rod, to_rod, from_rod);
}

int main()
{
	int n = 4;
	towerOfHanoi(n, 'A', 'C', 'B');	return 0;
}


/*
# Questions: Which is the base case of the algorithm?

# Answer:
 */