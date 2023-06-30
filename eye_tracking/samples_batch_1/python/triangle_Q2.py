# *************************************************************************

# The following code reasons about triangles in the geometrical sense.


class point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def square(x):
    return x * x

def order(a, b, c):
    copy = [a, b, c]
    copy.sort()
    return copy[0], copy[1], copy[2]

def euclidDistSquare(p1, p2):
    return square(p1.x - p2.x) + square(p1.y - p2.y)

def getSideClassification(a, b, c):
    if a == b and b == c:
        return "Equilateral"

    elif a == b or b == c:
        return "Isosceles"
    else:
        return "Scalene"

def getAngleClassification(a, b, c):

    if a + b > c:
        return "acute"

    elif a + b == c:
        return "right"
    else:
        return "obtuse"

def classifyTriangle(p1, p2, p3):

    a = euclidDistSquare(p1, p2)
    b = euclidDistSquare(p1, p3)
    c = euclidDistSquare(p2, p3)

    a, b, c = order(a, b, c)
    print("Triangle is ", getAngleClassification(a, b, c),
        " and ", getSideClassification(a, b, c))

p1 = point(3, 0)
p2 = point(0, 4)
p3 = point(4, 7)
classifyTriangle(p1, p2, p3)

p1 = point(0, 0)
p2 = point(1, 1)
p3 = point(1, 2)
classifyTriangle(p1, p2, p3)


# Questions: Which output will you get for the three points [1, 2], [1, 3],

# and [1, 4]?

# Answer: