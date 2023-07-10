import heapq

class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0  # Cost from start node to current node
        self.h = 0  # Heuristic (estimated) cost from current node to goal node
        self.f = 0  # Total cost (g + h)

    def __lt__(self, other):
        return self.f < other.f

def astar_search(graph, start, goal, heuristic):
    open_list = []
    closed_set = set()

    start_node = Node(start)
    goal_node = Node(goal)
    heapq.heappush(open_list, start_node)

    while open_list:
        current_node = heapq.heappop(open_list)
        closed_set.add(current_node.position)

        if current_node.position == goal_node.position:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent

            return path[::-1]

        neighbors = graph[current_node.position]

        for neighbor in neighbors:
            neighbor_node = Node(neighbor, current_node)
            if neighbor in closed_set:
                continue

            neighbor_node.g = current_node.g + 1
            neighbor_node.h = heuristic[neighbor]
            neighbor_node.f = neighbor_node.g + neighbor_node.h

            heapq.heappush(open_list, neighbor_node)

    return []

# Example usage:
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'C', 'D'],
    'C': ['A', 'B', 'D'],
    'D': ['B', 'C', 'E'],
    'E': ['D']
}

start = 'A'
goal = 'E'

heuristic = {
    'A': 4,
    'B': 2,
    'C': 3,
    'D': 1,
    'E': 0
}

path = astar_search(graph, start, goal, heuristic)

if path:
    print("Path found:")
    for node in path:
        print(node)
else:
    print("No path found.")
