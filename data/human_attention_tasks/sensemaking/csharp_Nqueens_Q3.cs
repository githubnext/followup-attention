/*************************************************************************************************/


/*
# The n-queens puzzle is the problem of placing n queens on an n x n chessboard

# such that no two queens attack each other.

# The algorithm below solves the following problem.
*/


using System;

class GFG
{
	readonly int N = 4;

	void printSolution(int [,]board)
	{
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < N; j++)
				Console.Write(" " + board[i, j]
								+ " ");
			Console.WriteLine();
		}
	}

	bool isSafe(int [,]board, int row, int col)
	{
		int i, j;

		for (i = 0; i < col; i++)
			if (board[row,i] == 1)
				return false;

		for (i = row, j = col; i >= 0 &&
			j >= 0; i--, j--)
			if (board[i,j] == 1)
				return false;

		for (i = row, j = col; j >= 0 &&
					i < N; i++, j--)
			if (board[i, j] == 1)
				return false;

		return true;
	}

	bool solveNQUtil(int [,]board, int col)
	{
		if (col >= N)
			return true;

		for (int i = 0; i < N; i++)
		{
			if (isSafe(board, i, col))
			{
				board[i, col] = 1;

				if (solveNQUtil(board, col + 1) == true)
					return true;

				board[i, col] = 0;
			}
		}

		return false;
	}

	bool solveNQ()
	{
		int [,]board = {{ 0, 0, 0, 0 },
						{ 0, 0, 0, 0 },
						{ 0, 0, 0, 0 },
						{ 0, 0, 0, 0 }};

		if (solveNQUtil(board, 0) == false)
		{
			Console.Write("Solution does not exist");
			return false;
		}

		printSolution(board);
		return true;
	}

	public static void Main(String []args)
	{
		GFG Queen = new GFG();
		Queen.solveNQ();
	}
}

/*
# Question: How would you expect the run time of `solveNQ(n)` to scale with `n`?

# Answer:
*/