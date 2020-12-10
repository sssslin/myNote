package basic.skill.list;


import test.ListNode;

/**
 * 理解链表的指针或者说引用的方法就是多画图。
 */
public class ListBasicSkill {

    // 删除node后面的一个结点（假设node为链表中的一个非尾部结点）
    public static void deleteUnTailNode(ListNode node) {

        node.next = node.next.next;
    }

    // 往指定节点后插入一个结点
    public static void insertNode(ListNode node) {

        ListNode x = new ListNode(0);
        // node节点先指向x的后续节点
        node.next = x.next;
        // x节点指向t节点
        x.next = node;
    }

    // 往链表头添加元素
    public static void addNodeInHead(ListNode node) {
        ListNode dummy = new ListNode(0);
        dummy.next = node;
        node = dummy;
    }



}
