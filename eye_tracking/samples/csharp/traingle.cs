using System;

class GFG
{
    public class point
    {
        public int x, y;
        public point() { }

        public point(int x, int y)
        {
            this.x = x;
            this.y = y;
        }
    };

    static int square(int x)
    {
        return x * x;
    }
    static int a, b, c;

    static void order()
    {
        int[] copy = new int[3];
        copy[0] = a;
        copy[1] = b;
        copy[2] = c;
        Array.Sort(copy);
        a = copy[0];
        b = copy[1];
        c = copy[2];
    }

    static int euclidDistSquare(point p1,
                                point p2)
    {
        return square(p1.x - p2.x) +
            square(p1.y - p2.y);
    }

    static String getSideClassification(int a,
                                        int b, int c)
    {
        if (a == b && b == c)
            return "Equilateral";

        else if (a == b || b == c)
            return "Isosceles";

        else
            return "Scalene";
    }

    static String getAngleClassification(int a,
                                        int b, int c)
    {
        if (a + b > c)
            return "acute";

        else if (a + b == c)
            return "right";

        else
            return "obtuse";
    }

    static void classifyTriangle(point p1,
                                point p2,
                                point p3)
    {
        a = euclidDistSquare(p1, p2);
        b = euclidDistSquare(p1, p3);
        c = euclidDistSquare(p2, p3);

        order();

        Console.WriteLine("Triangle is "
                    + getAngleClassification(a, b, c)
                    + " and "
                    + getSideClassification(a, b, c));
    }

    public static void Main(String[] args)
    {
        point p1, p2, p3;
        p1 = new point(3, 0);
        p2 = new point(0, 4);
        p3 = new point(4, 7);
        classifyTriangle(p1, p2, p3);

        p1 = new point(0, 0);
        p2 = new point(1, 1);
        p3 = new point(1, 2);
        classifyTriangle(p1, p2, p3);
    }
}