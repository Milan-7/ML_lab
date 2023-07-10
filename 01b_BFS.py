from queue import PriorityQueue

def best_first_search(graph, start, goal, heuristic):
    visited = set()
    pq = PriorityQueue()
    pq.put((heuristic[start], start))
    
    while not pq.empty():
        _, node = pq.get()
        
        if node == goal:
            print("Goal reached!")
            return
        
        if node not in visited:
            print("Visiting node:", node)
            visited.add(node)
            
            for neighbor, _ in graph[node]:
                if neighbor not in visited:
                    pq.put((heuristic[neighbor], neighbor))
    
    print("Goal not found!")

# Example graph representation using adjacency list
graph = {
    'A': [('B', 4), ('C', 2)],
    'B': [('A', 4), ('D', 5)],
    'C': [('A', 2), ('E', 3)],
    'D': [('B', 5), ('E', 1)],
    'E': [('C', 3), ('D', 1)]
}

start_node = 'A'
goal_node = 'E'

#Heuristic values from curr node -> goal node
heuristic_values = {
    'A': 5,
    'B': 3,
    'C': 2,
    'D': 4,
    'E': 0
}

best_first_search(graph, start_node, goal_node, heuristic_values)
