/*************************************************************************************************/


/*
# In the Tower of Hanoi problem you have to move all the disks

# from the left hand post to the right hand post.

# You can only move the disks one at a time and you can never

# place a bigger disk on a smaller disk.
 */


using System;

class GFG
{

    static void towerOfHanoi(int n, char from_rod, char to_rod, char aux_rod)
    {

        if (n == 0)
        {
            return;
        }

        towerOfHanoi(n-1, from_rod, aux_rod, to_rod);

        Console.WriteLine("Move disk " + n + " from rod " +
                          from_rod + " to rod " + to_rod);

        towerOfHanoi(n-1, aux_rod, to_rod, from_rod);
    }

    public static void Main(String []args)
    {
        int n = 4;
        towerOfHanoi(n, 'A', 'C', 'B');
    }
}


/*
# Questions: which is the name of the auxiliary rod in the call

# TowerOfHanoi(n, 'Mark', 'Mat', 'Luke')?

# Answer:
 */
