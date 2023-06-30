/*************************************************************************************************/
using System;

class GFG
{
    public static int preindex;

    public class node
    {
        public int data;
        public node left, right;

        public node(int data)
        {
            this.data = data;
        }
    }

    public static node constructTreeUtil(int[] pre, int[] post,
                                        int l, int h, int size)
    {

        if (preindex >= size || l > h)
        {
            return null;
        }

        node root = new node(pre[preindex]);
        preindex++;

        if (l == h || preindex >= size)
        {
            return root;
        }
        int i;

        for (i = l; i <= h; i++)
        {
            if (post[i] == pre[preindex])
            {
                break;
            }
        }

        if (i <= h)
        {
            root.left = constructTreeUtil(pre, post,
                                        l, i, size);
            root.right = constructTreeUtil(pre, post,
                                        i + 1, h - 1, size);
        }
        return root;
    }

    public static node constructTree(int[] pre,
                                    int[] post, int size)
    {
        preindex = 0;
        return constructTreeUtil(pre, post, 0, size - 1, size);
    }

    public static void printInorder(node root)
    {
        if (root == null)
        {
            return;
        }
        printInorder(root.left);
        Console.Write(root.data + " ");
        printInorder(root.right);
    }

    // Driver Code
    public static void Main(string[] args)
    {
        int[] pre = new int[] { 1, 2, 4, 8, 9, 5, 3, 6, 7 };
        int[] post = new int[] { 8, 9, 4, 5, 2, 6, 7, 3, 1 };

        int size = pre.Length;
        node root = constructTree(pre, post, size);

        Console.WriteLine("Inorder traversal of " +
                        "the constructed tree:");
        printInorder(root);
    }
}