

## Tree

Adding the following comment at the end of the file after the main, we got:

1. How many calls to `constructTreeUtil` will `constructTree([1, 2, 3], [ 1, 2, 3], 2)` make?

```python
# Question: How many calls to `constructTreeUtil` will `constructTree([1, 2, 3], [ 1, 2, 3], 2)` make?
# Answer: 1
```

2. Under which conditions could the check `if i <= h` in `constructTreeUtil` be false?

```python
# Question: Under which conditions could the check `if i <= h` in `constructTreeUtil` be false?
# Answer: When the value of `pre[preIndex]` is not equal to the value of `post[i]`.
```

3. A part of the code you don't have direct access to has called `constructTree` with unknown parameters. What can you find out about those parameters?

```python
# Question: A part of the code you don't have direct access to has called `constructTree` with unknown parameters. What can you find out about those parameters?
# Answer: The parameters are `pre` and `post` which are the preorder and postorder traversals of the tree.
```


## Hanoi

1. How does the algorithm moves disks from the starting rod to the ending rod?

```python
# Questions: How does the algorithm moves disks from the starting rod to the ending rod?
# Answer: The algorithm moves disks from the starting rod to the auxiliary rod, then the auxiliary rod to the ending rod.
```

2. Which is the base case of the algorithm?

```python
# Questions: Which is the base case of the algorithm?
# Answer: The base case is when n == 0.
```

3. Which is the name of the auxiliary rod in the call TowerOfHanoi(n, 'Marco', 'Matteo', 'Luca')?

```python
# Questions: which is the name of the auxiliary rod in the call TowerOfHanoi(n, 'Marco', 'Matteo', 'Luca')?
# Answer: B
```


##  NQueens

What about adding some context for human and model?
```python
#The n-queens puzzle is the problem of placing n queens on an n x n chessboard such that no two queens attack each other.

#Given an integer n, return all distinct solutions to the n-queens puzzle. You may return the answer in any order.

#Each solution contains a distinct board configuration of the n-queens' placement, where 1 and 0 both indicate a queen and an empty space, respectively.
```
Taken and adapted from this: https://leetcode.com/problems/n-queens/

1. What does `solveNQ(-13)` return?

```python
# Question: What does `solveNQ(-13)` return?
# Answer: -13
```

2. What are valid dimensions and values for the array `board`?

```python
# Question: What are valid dimensions and values for the array `board`?
# Answer: The array `board` has 4 rows and 4 columns. The values in the array are either 0 or 1.
```

3. How would you expect the run time of `solveNQ(n)` to scale with `n`?

```python
# Question: How would you expect the run time of `solveNQ(n)` to scale with `n`?
# Answer: O(n^2)
```

## Triangle

1. Which of the functions have side effects?

```python
# Questions: Which of the functions have side effects?
# Answer: None of them.
```

2. Which output will you get for the three points [1, 2], [1, 3], and [1, 4]?

```python
# Questions: Which output will you get for the three points [1, 2], [1, 3], and [1, 4]?
# Answer: Right
```

3. What could happen if the call to `order()` were omitted from `classifyTriangle`?

```python
# Question: What could happen if the call to `order()` were omitted from `classifyTriangle`?
# Answer: The program would still work, but the order of the sides would be undefined.
```


## Scheduler

1. You have a fresh instance of a `ScheduledTask` making a time intensive call. How would you assign the result of the call to a variable?
3. What is the role of the parameter expirationMs?
4. How does the `ScheduledTask` class deal with errors in the call it wraps?

```python
# Note: If we have a fresh instance of a ScheduledTask making a time intensive call,
# the way to assign the result of the call to a variable is:
#   task = ScheduledTask(...)
#   result = task.run()
# If you set expirationMs to a number, the result will be None after the expiration time.
# If the call throws an error, the result will be None.
```

(Copilot completed those. Its first answer forgets to await. I still count it as okay enough, especially seeing that it did await when the code looked slightly different.)
