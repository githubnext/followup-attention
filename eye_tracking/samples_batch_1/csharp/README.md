# Sources for those files

* scheduler.cs: @wunderalbert translated Copilot code to C#.

# Tasks for these files:

## Summarize this code

"Explain what this code does as if you would explain it to a colleague."

## Sense making questions

* Nqueens.cs
  1. What does `solveNQ(-13)` return? 
  2. What are valid dimensions and values for the array `board`?
  3. How would you expect the run time of `solveNQ(n)` to scale with `n`?
* scheduler.cs
  1. You have a fresh instance of a `ScheduledTask` making a time intensive call. How would you assign the result of the call to a variable?
  2. What is the role of the parameter expirationMs?
  3. How does the `ScheduledTask` class deal with errors in the call it wraps?
* hanoi.cs -- plan to scrap
* triangle.cs
  1. Which of the functions have side effects?
  2. Which output will you get for the three points [1, 2], [1, 3], and [1, 4]?
  3. What could happen if the call to `order()` were omitted from `classifyTriangle`?
* tree.cs
  1. How many calls to `constructTreeUtil` will `constructTree( new [] { 1, 2, 3 },  new [] { 1, 2, 3  }, 2)` make?
  2. Under which conditions could the check `if (i <= h)` in `constructTreeUtil` be false?
  3. A part of the code you don't have direct access to has called `constructTree` with unknown parameters. What can you find out about those parameters?

## Write unit tests

"Write one or two unit tests for relevant aspects of this code. Feel free to skip test framework boilerplate."
