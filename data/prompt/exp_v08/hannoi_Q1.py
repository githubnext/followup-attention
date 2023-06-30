# ****************************************************************************

# In the Tower of Hanoi problem you have to move all the disks

# from the left hand post to the right hand post.

# You can only move the disks one at a time and you can never

# place a bigger disk on a smaller disk.


def TowerOfHanoi(n, from_rod, to_rod, aux_rod):
    if n == 0:
        return
    TowerOfHanoi(n-1, from_rod, aux_rod, to_rod)
    print("Move disk", n, "from rod", from_rod, "to rod", to_rod)
    TowerOfHanoi(n-1, aux_rod, to_rod, from_rod)


n = 4
TowerOfHanoi(n, 'A', 'C', 'B')


# Questions: How does the algorithm moves disks from the starting rod to the

# ending rod?

# Answer:
