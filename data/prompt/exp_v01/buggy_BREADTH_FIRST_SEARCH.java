package java_programs;
import java.util.*;
import java.util.ArrayDeque;
import java_programs.Node;

public class BREADTH_FIRST_SEARCH {

    public static Set<Node> nodesvisited = new HashSet<>();

    public static boolean breadth_first_search(Node startnode, Node goalnode) {
        Deque<Node> queue = new ArrayDeque<>();
        queue.addLast(startnode);

        nodesvisited.add(startnode);

#**#STOP#**#
        while (true) {
            Node node = queue.removeFirst();

            if (node == goalnode) {
                return true;
            } else {
                for (Node successor_node : node.getSuccessors()) {
                    if (!nodesvisited.contains(successor_node)) {
                        queue.addFirst(successor_node);
                        nodesvisited.add(successor_node);
                    }
                }
            }
        }
        /**
         * The buggy program always drops into while(true) loop and will not return false
         * Removed below line to fix compilation error
         */
        // return false;
    }

}