/***************************************************************************/

/*
# The following code reasons about triangles in the geometrical sense.
*/


#include<iostream>
#include <bits/stdc++.h>
using namespace std;

struct point {
	int x, y;
	point() {}
	point(int x, int y)
		: x(x), y(y)
	{
	}
};

int square(int x)
{
	return x * x;
}

void order(int& a, int& b, int& c)
{
	int copy[3];
	copy[0] = a;
	copy[1] = b;
	copy[2] = c;
	sort(copy, copy + 3);
	a = copy[0];
	b = copy[1];
	c = copy[2];
}

int euclidDistSquare(point p1, point p2)
{
	return square(p1.x - p2.x) + square(p1.y - p2.y);
}

string getSideClassification(int a, int b, int c)
{
	if (a == b && b == c)
		return "Equilateral";

	else if (a == b || b == c)
		return "Isosceles";

	else
		return "Scalene";
}

string getAngleClassification(int a, int b, int c)
{
	if (a + b > c)
		return "acute";

	else if (a + b == c)
		return "right";

	else
		return "obtuse";
}

void classifyTriangle(point p1, point p2, point p3)
{
	int a = euclidDistSquare(p1, p2);
	int b = euclidDistSquare(p1, p3);
	int c = euclidDistSquare(p2, p3);

	order(a, b, c);

	cout << "Triangle is "
				+ getAngleClassification(a, b, c)
				+ " and "
				+ getSideClassification(a, b, c)
		<< endl;
}

int main()
{
	point p1, p2, p3;
	p1 = point(3, 0);
	p2 = point(0, 4);
	p3 = point(4, 7);
	classifyTriangle(p1, p2, p3);

	p1 = point(0, 0);
	p2 = point(1, 1);
	p3 = point(1, 2);
	classifyTriangle(p1, p2, p3);
	return 0;
}

/*
# Questions: Which output will you get for the three points [1, 2], [1, 3],

# and [1, 4]?

# Answer:
*/