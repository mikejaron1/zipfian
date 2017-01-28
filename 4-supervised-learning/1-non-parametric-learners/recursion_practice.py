from node import TreeNode


def sum_digits(num):
    '''
    INPUT: int
    OUTPUT: int

    Return the sum of the digits of the integer num.
    '''
    if num == 0:
        return 0
    return num % 10 + sum_digits(num / 10)


def sum_tree(root):
    '''
    INPUT: TreeNode
    OUTPUT: int

    Return the sum of all the values in the binary tree with the given root.
    '''
    if not root:
        return 0
    return root.value + sum_tree(root.left) + sum_tree(root.right)


# Three versions of print_all for preorder, postorder, inorder
def print_all_preorder(root):
    '''
    INPUT: TreeNode
    OUTPUT: None

    Print all the values in the tree rooted at the given root (in any order).
    '''
    if root:
        print root.value
        print_all(root.left)
        print_all(root.right)


def print_all_postorder(root):
    '''
    INPUT: TreeNode
    OUTPUT: None

    Print all the values in the tree rooted at the given root (in any order).
    '''
    if root:
        print_all(root.left)
        print_all(root.right)
        print root.value


def print_all_inorder(root):
    '''
    INPUT: TreeNode
    OUTPUT: None

    Print all the values in the tree rooted at the given root (in any order).
    '''
    if root:
        print_all(root.left)
        print root.value
        print_all(root.right)


def build_coinflip_tree(k, value=""):
    '''
    INPUT: int
    OUTPUT: TreeNode

    Return the root of a binary tree representing all the possible outcomes
    form flipping a coin k times.
    '''
    node = TreeNode(value)
    if k != 0:
        node.left = build_coinflip_tree(k - 1, value + "H")
        node.right = build_coinflip_tree(k - 1, value + "T")
    return node


## EXTRA CREDIT

def build_from_traversals(preorder, inorder):
    if not preorder or not inorder:
        return None
    root_val = preorder[0]
    i = inorder.find(root_val)
    root = TreeNode(root_val)
    root.left = build_from_traversals(preorder[1:i+1], inorder[0:i])
    root.right = build_from_traversals(preorder[i+1:], inorder[i+1:])


def print_tree(root, level=0):
    if not root:
        return
    indent = "   |" * level + "-> "
    print indent + str(root.value)
    print_tree(root.left, level + 1)
    print_tree(root.right, level + 1)


def make_word_breaks(phrase, word_list):
    if not phrase:
        return []
    for i in xrange(1, len(phrase) + 1):
        word = phrase[:i]
        if word in word_list:
            rest = make_word_breaks(phrase[i:], word_list)
            if rest is not None:
                return [word] + rest
    return None

