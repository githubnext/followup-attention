#*************************************************************************************************#

class Node:

    def __init__(self, data):

        self.data = data
        self.left = None
        self.right = None


def constructTreeUtil(pre: list, post: list,
                      l: int, h: int,
                      size: int) -> Node:
    global preIndex

    if (preIndex >= size or l > h):
        return None

    root = Node(pre[preIndex])
    preIndex += 1

    if (l == h or preIndex >= size):
        return root

    i = l
    while i <= h:
        if (pre[preIndex] == post[i]):
            break

        i += 1

    if (i <= h):
        root.left = constructTreeUtil(pre, post,
                                      l, i, size)
        root.right = constructTreeUtil(pre, post,
                                       i + 1, h-1,
                                       size)

    return root


def constructTree(pre: list,
                  post: list,
                  size: int) -> Node:

    global preIndex

    return constructTreeUtil(pre, post, 0,
                             size - 1, size)


def printInorder(node: Node) -> None:

    if (node is None):
        return

    printInorder(node.left)
    print(node.data, end=" ")

    printInorder(node.right)


if __name__ == "__main__":

    pre = [1, 2, 4, 8, 9, 5, 3, 6, 7]
    post = [8, 9, 4, 5, 2, 6, 7, 3, 1]
    size = len(pre)

    preIndex = 0

    root = constructTree(pre, post, size)

    print("Inorder traversal of "
          "the constructed tree: ")

    printInorder(root)
