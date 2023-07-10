class Tree:
    def __init__(self, value, children=[]):
        self.value = value
        self.children = children


# Define the game tree using the Tree data structure
game_tree = Tree(0, [Tree(0, [Tree(3), Tree(12)]), Tree(0, [Tree(8), Tree(2)])])


def minimax(node, depth, alpha, beta, maximizing_player):
    if depth == 0 or not node.children:
        return node.value, [node.value]

    if maximizing_player:
        max_value = float("-inf")
        max_path = []
        for child_node in node.children:
            child_value, child_path = minimax(child_node, depth - 1, alpha, beta, False)
            if child_value > max_value:
                max_value = child_value
                max_path = [node.value] + child_path
            alpha = max(alpha, max_value)
            if alpha >= beta:
                break
        return max_value, max_path
    else:
        min_value = float("inf")
        min_path = []
        for child_node in node.children:
            child_value, child_path = minimax(child_node, depth - 1, alpha, beta, True)
            if child_value < min_value:
                min_value = child_value
                min_path = [node.value] + child_path
            beta = min(beta, min_value)
            if alpha >= beta:
                break
        return min_value, min_path


# Example usage:
optimal_value, optimal_path = minimax(game_tree, 2, float("-inf"), float("inf"), True)
print("Optimal value:", optimal_value)
print("Optimal path:", optimal_path)
