import time
import heapq
from collections import deque
import networkx as nx
import matplotlib.pyplot as plt
import random
import ipywidgets as widgets
from IPython.display import display, clear_output

def create_demo_graph():
    """
    Create a weighted graph with an interesting structure for demonstrating search algorithms.
    Returns a NetworkX graph object.
    """
    G = nx.Graph()

    # Add nodes
    nodes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M']
    G.add_nodes_from(nodes)

    # Add edges with weights
    edges = [
        ('A', 'B', 4), ('A', 'C', 3), ('B', 'E', 12), ('B', 'F', 5),
        ('C', 'D', 7), ('C', 'G', 10), ('D', 'H', 2), ('D', 'I', 4),
        ('E', 'J', 5), ('F', 'J', 8), ('F', 'K', 6), ('G', 'K', 8),
        ('G', 'L', 7), ('H', 'L', 9), ('I', 'L', 6), ('I', 'M', 5),
        ('J', 'M', 10), ('K', 'M', 7), ('L', 'M', 3)
    ]

    G.add_weighted_edges_from(edges)

    # Add positions for visualization
    pos = {
        'A': (0, 0), 'B': (-1, -1), 'C': (1, -1), 'D': (2, -2),
        'E': (-2, -2), 'F': (-1, -2), 'G': (1, -2), 'H': (2, -3),
        'I': (3, -3), 'J': (-2, -3), 'K': (0, -3), 'L': (2, -4),
        'M': (0, -5)
    }

    nx.set_node_attributes(G, pos, 'pos')
    return G

def create_romania_map():
    """
    Create the Romania map from the AIMA textbook with real city names and distances.
    Returns a NetworkX graph object.
    """
    G = nx.Graph()

    # Add nodes (cities)
    cities = [
        'Arad', 'Bucharest', 'Craiova', 'Drobeta', 'Eforie', 'Fagaras',
        'Giurgiu', 'Hirsova', 'Iasi', 'Lugoj', 'Mehadia',
        'Neamt', 'Oradea', 'Pitesti', 'Rimnicu Vilcea', 'Sibiu',
        'Timisoara', 'Urziceni', 'Vaslui', 'Zerind'
    ]
    G.add_nodes_from(cities)

    # Add edges with weights (distances between cities in km)
    edges = [
        ('Arad', 'Zerind', 75), ('Arad', 'Sibiu', 140), ('Arad', 'Timisoara', 118),
        ('Bucharest', 'Fagaras', 211), ('Bucharest', 'Pitesti', 101),
        ('Bucharest', 'Giurgiu', 90), ('Bucharest', 'Urziceni', 85),
        ('Craiova', 'Drobeta', 120), ('Craiova', 'Rimnicu Vilcea', 146), ('Craiova', 'Pitesti', 138),
        ('Drobeta', 'Mehadia', 75),
        ('Eforie', 'Hirsova', 86),
        ('Fagaras', 'Sibiu', 99),
        ('Hirsova', 'Urziceni', 98),
        ('Iasi', 'Neamt', 87), ('Iasi', 'Vaslui', 92),
        ('Lugoj', 'Timisoara', 111), ('Lugoj', 'Mehadia', 70),
        ('Oradea', 'Zerind', 71), ('Oradea', 'Sibiu', 151),
        ('Pitesti', 'Rimnicu Vilcea', 97),
        ('Rimnicu Vilcea', 'Sibiu', 80),
        ('Urziceni', 'Vaslui', 142)
    ]

    G.add_weighted_edges_from(edges)

    # Add positions for visualization (approximate geographical positions)
    pos = {
        'Arad': (1, 4),
        'Zerind': (1, 5),
        'Oradea': (1, 6),
        'Timisoara': (1, 2),
        'Lugoj': (2, 2),
        'Mehadia': (2, 1),
        'Drobeta': (2, 0),
        'Craiova': (4, 0),
        'Sibiu': (3, 4),
        'Fagaras': (5, 4),
        'Rimnicu Vilcea': (4, 3),
        'Pitesti': (5, 2),
        'Bucharest': (6, 1),
        'Giurgiu': (6, 0),
        'Urziceni': (7, 2),
        'Hirsova': (9, 2),
        'Eforie': (9, 0),
        'Vaslui': (8, 3),
        'Iasi': (8, 4),
        'Neamt': (7, 5)
    }

    nx.set_node_attributes(G, pos, 'pos')
    return G

def visualize_graph(G, title="Graph", highlighted_path=None, highlighted_nodes=None, frontier=None):
    """
    Visualize a graph with optional highlighted path, nodes, and frontier.

    Args:
        G: NetworkX graph
        title: Title for the plot
        highlighted_path: List of edges to highlight (path)
        highlighted_nodes: List of nodes to highlight (visited)
        frontier: List of nodes in the frontier
    """
    plt.figure(figsize=(6, 6))  # Make figure square
    pos = nx.get_node_attributes(G, 'pos')

    # Draw the basic graph
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

    # Draw edges with weights
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)

    # Highlight visited nodes if provided
    if highlighted_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=highlighted_nodes, node_color='yellow', node_size=500)

    # Highlight frontier nodes if provided
    if frontier:
        nx.draw_networkx_nodes(G, pos, nodelist=frontier, node_color='orange', node_size=500)

    # Highlight path if provided
    if highlighted_path:
        path_edges = [(highlighted_path[i], highlighted_path[i+1]) for i in range(len(highlighted_path)-1)]
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=3, edge_color='red')

    # Add coordinate grid
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xticks(range(int(min(x for x, _ in pos.values())) - 1, int(max(x for x, _ in pos.values())) + 2))
    plt.yticks(range(int(min(y for _, y in pos.values())) - 1, int(max(y for _, y in pos.values())) + 2))

    # Set aspect ratio to be equal for square cells
    plt.gca().set_aspect('equal')

    plt.title(title, fontsize=14)
    plt.tight_layout()
    return plt.gcf()

def graph_search(G, start, goal, algorithm='bfs', heuristic=None):
    """
    Generic search function that can perform various search algorithms on a graph.

    Args:
        G: NetworkX graph
        start: Starting node
        goal: Goal node
        algorithm: Search algorithm to use ('bfs', 'dfs', 'ucs', 'greedy', 'astar')
        heuristic: Heuristic function for informed search (required for 'greedy' and 'astar')

    Returns:
        path: List of nodes in the path from start to goal
        visited_order: List of visited nodes in order
        metrics: Dictionary with metrics (time, space, path_cost)
        states: Dictionary with states for visualization (frontier at each step, etc.)
    """
    start_time = time.time()

    if algorithm == 'bfs':
        frontier = deque([(start, [start], 0)])  # (node, path, cost)
    elif algorithm == 'dfs':
        frontier = [(start, [start], 0)]  # Using list as stack
    elif algorithm in ['ucs', 'greedy', 'astar']:
        # For priority queue: (priority, node, path, cost)
        # Priority depends on the algorithm
        if algorithm == 'ucs':
            frontier = [(0, start, [start], 0)]  # Priority is cost
        elif algorithm == 'greedy':
            if not heuristic:
                raise ValueError("Heuristic function required for greedy search")
            frontier = [(heuristic(start, goal), start, [start], 0)]  # Priority is heuristic
        elif algorithm == 'astar':
            if not heuristic:
                raise ValueError("Heuristic function required for A* search")
            frontier = [(heuristic(start, goal), start, [start], 0)]  # Priority is f(n) = g(n) + h(n)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    visited = set()
    visited_order = []
    max_frontier_size = 1

    # For visualization
    states = {
        'steps': []
    }

    while frontier:
        max_frontier_size = max(max_frontier_size, len(frontier))

        if algorithm == 'bfs':
            current, path, cost = frontier.popleft()
        elif algorithm == 'dfs':
            current, path, cost = frontier.pop()
        elif algorithm in ['ucs', 'greedy', 'astar']:
            priority, current, path, cost = heapq.heappop(frontier)

        # Save state for visualization
        frontier_nodes = []
        if algorithm == 'bfs':
            frontier_nodes = [node for node, _, _ in frontier]
        elif algorithm == 'dfs':
            frontier_nodes = [node for node, _, _ in frontier]
        elif algorithm in ['ucs', 'greedy', 'astar']:
            frontier_nodes = [node for _, node, _, _ in frontier]

        if current not in visited:
            visited.add(current)
            visited_order.append(current)

            states['steps'].append({
                'current': current,
                'path': path.copy(),
                'visited': list(visited),
                'frontier': frontier_nodes.copy(),
                'cost': cost
            })

            if current == goal:
                end_time = time.time()
                metrics = {
                    'time': end_time - start_time,
                    'space': max_frontier_size,
                    'path_cost': cost,
                    'nodes_visited': len(visited)
                }
                return path, visited_order, metrics, states

            # Expand current node
            for neighbor in G.neighbors(current):
                if neighbor not in visited:
                    new_path = path + [neighbor]
                    edge_cost = G[current][neighbor]['weight']
                    new_cost = cost + edge_cost

                    if algorithm == 'bfs':
                        frontier.append((neighbor, new_path, new_cost))
                    elif algorithm == 'dfs':
                        frontier.append((neighbor, new_path, new_cost))
                    elif algorithm == 'ucs':
                        heapq.heappush(frontier, (new_cost, neighbor, new_path, new_cost))
                    elif algorithm == 'greedy':
                        h = heuristic(neighbor, goal)
                        heapq.heappush(frontier, (h, neighbor, new_path, new_cost))
                    elif algorithm == 'astar':
                        h = heuristic(neighbor, goal)
                        f = new_cost + h  # f(n) = g(n) + h(n)
                        heapq.heappush(frontier, (f, neighbor, new_path, new_cost))

    # If no path is found
    end_time = time.time()
    metrics = {
        'time': end_time - start_time,
        'space': max_frontier_size,
        'path_cost': float('inf'),
        'nodes_visited': len(visited)
    }
    return None, visited_order, metrics, states

# Define heuristic functions for the graph
def euclidean_distance_heuristic(node1, node2, G):
    """
    Euclidean distance heuristic based on node positions.
    """
    pos = nx.get_node_attributes(G, 'pos')
    x1, y1 = pos[node1]
    x2, y2 = pos[node2]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

def manhattan_distance_heuristic(node1, node2, G):
    """
    Manhattan distance heuristic based on node positions.
    """
    pos = nx.get_node_attributes(G, 'pos')
    x1, y1 = pos[node1]
    x2, y2 = pos[node2]
    return abs(x1 - x2) + abs(y1 - y2)


def run_graph_search_example(graph, algo, start, goal, heuristic="None"):
    """
    Run a specific search algorithm on a graph and display the results.

    Args:
        graph: NetworkX graph
        algo: Search algorithm to use ('bfs', 'dfs', 'ucs', 'greedy', 'astar')
        start: Starting node
        goal: Goal node
        heuristic: Heuristic type as string ('None', 'euclidean', 'manhattan')

    Returns:
        path: List of nodes in the solution path
        visited: List of visited nodes
        metrics: Dictionary with metrics (time, space, path_cost)
    """
    import matplotlib.pyplot as plt

    print(f"\nRunning {algo.upper()} search from {start} to {goal}...")

    # Convert heuristic string to function
    heuristic_function = None
    if heuristic.lower() == "euclidean":
        heuristic_function = lambda node1, node2: euclidean_distance_heuristic(node1, node2, graph)
    elif heuristic.lower() == "manhattan":
        heuristic_function = lambda node1, node2: manhattan_distance_heuristic(node1, node2, graph)

    # For greedy and A*, ensure we have a heuristic
    if algo.lower() in ['greedy', 'astar'] and heuristic_function is None:
        print(f"Warning: {algo} requires a heuristic. Using euclidean distance as default.")
        heuristic_function = lambda node1, node2: euclidean_distance_heuristic(node1, node2, graph)

    # Run the search
    path, visited, metrics, states = graph_search(
        graph, start, goal,
        algorithm=algo.lower(),
        heuristic=heuristic_function
    )

    # Display results
    if path:
        print(f"✓ Path found! Length: {len(path) - 1}, Cost: {metrics['path_cost']}")
        print(f"Path: {' -> '.join(str(node) for node in path)}")
    else:
        print("✗ No path found!")

    print(f"Nodes visited: {metrics['nodes_visited']}")
    print(f"Max frontier size: {metrics['space']}")
    print(f"Time taken: {metrics['time']:.6f} seconds")

    fig = visualize_graph(
        graph,
        title=f"{algo.upper()} Search Result",
        highlighted_path=path,
        highlighted_nodes=visited
    )
    plt.show()

    return path, visited, metrics

def create_graph_search_demo(demo_graph):
    """
    Create a demo of the graph search algorithms.
    """
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    import matplotlib.pyplot as plt

    # Function to run a search algorithm and collect step-by-step states
    def run_graph_search_demo(algorithm, start_node, goal_node, heuristic_name):
        # Convert heuristic name to lowercase string for consistency with main function
        heuristic_str = 'None'
        if heuristic_name == 'Euclidean':
            heuristic_str = 'euclidean'
        elif heuristic_name == 'Manhattan':
            heuristic_str = 'manhattan'

        # Select appropriate heuristic
        heuristic = None
        if heuristic_str == 'euclidean':
            def heuristic(node1, node2): return euclidean_distance_heuristic(
                node1, node2, demo_graph)
        elif heuristic_str == 'manhattan':
            def heuristic(node1, node2): return manhattan_distance_heuristic(
                node1, node2, demo_graph)

        # Run the search algorithm
        path, visited, metrics, states = graph_search(
            demo_graph, start_node, goal_node,
            algorithm=algorithm.lower(),
            heuristic=heuristic
        )

        return path, visited, metrics, states

    # Get list of nodes for dropdowns
    nodes = list(demo_graph.nodes())

    # Choose appropriate defaults based on available nodes
    default_start = nodes[0]  # First node as default start
    default_goal = nodes[-1]  # Last node as default goal

    # Create interactive widgets
    algorithm_dropdown = widgets.Dropdown(
        options=['BFS', 'DFS', 'UCS', 'Greedy', 'Astar'],
        value='BFS',
        description='Algorithm:',
        style={'description_width': 'initial'}
    )

    start_node_dropdown = widgets.Dropdown(
        options=nodes,
        value=default_start,
        description='Start Node:',
        style={'description_width': 'initial'}
    )

    goal_node_dropdown = widgets.Dropdown(
        options=nodes,
        value=default_goal,
        description='Goal Node:',
        style={'description_width': 'initial'}
    )

    heuristic_dropdown = widgets.Dropdown(
        options=['Euclidean', 'Manhattan'],
        value='Euclidean',
        description='Heuristic:',
        style={'description_width': 'initial'}
    )

    step_slider = widgets.IntSlider(
        min=0,
        max=1,
        step=1,
        value=0,
        description='Step:',
        style={'description_width': 'initial'},
        continuous_update=False
    )

    run_button = widgets.Button(
        description='Run Search',
        button_style='success',
        icon='play'
    )

    next_step_button = widgets.Button(
        description='Next Step',
        button_style='info',
        icon='arrow-right'
    )

    output = widgets.Output()
    graph_output = widgets.Output()
    info_output = widgets.Output()

    # Store search results
    search_results = {}

    def update_graph(step):
        with graph_output:
            clear_output(wait=True)
            if not search_results:
                fig = visualize_graph(
                    demo_graph, title="Demo Graph (Run search first)")
                plt.show()
                return

            states = search_results.get('states')
            if not states or not states['steps'] or step >= len(states['steps']):
                return

            current_state = states['steps'][step]
            current = current_state['current']
            path = current_state['path']
            visited = current_state['visited']
            frontier = current_state['frontier']

            title = f"Step {step+1}/{len(states['steps'])}: {algorithm_dropdown.value} Search"
            fig = visualize_graph(
                demo_graph,
                title=title,
                highlighted_path=path,
                highlighted_nodes=visited,
                frontier=frontier
            )
            plt.show()

            # Display metadata below the graph
            if not search_results:
                print("Run search first to see step information.")
                return

            if not states or not states['steps'] or step >= len(states['steps']):
                return

            current_state = states['steps'][step]
            current = current_state['current']
            path = current_state['path']
            cost = current_state['cost']

            print(f"Current node: {current}")
            print(f"Path so far: {' -> '.join(path)}")
            print(f"Path cost: {cost}")
            print(f"Visited nodes: {len(current_state['visited'])}")
            print(f"Frontier size: {len(current_state['frontier'])}")

            # If we found the goal
            if current == goal_node_dropdown.value:
                print("\n🎯 Goal reached!")
                metrics = search_results.get('metrics')
                if metrics:
                    print(f"\nSearch Metrics:")
                    print(f"Time taken: {metrics['time']:.6f} seconds")
                    print(f"Max frontier size: {metrics['space']}")
                    print(f"Total nodes visited: {metrics['nodes_visited']}")

    def on_step_change(change):
        if change['name'] == 'value':
            update_graph(change['new'])

    def on_run_button_clicked(b):
        with output:
            clear_output()
            algorithm = algorithm_dropdown.value
            start = start_node_dropdown.value
            goal = goal_node_dropdown.value
            heuristic = heuristic_dropdown.value

            print(f"Running {algorithm} search from {start} to {goal}...")

            # Run the search
            path, visited, metrics, states = run_graph_search_demo(
                algorithm, start, goal, heuristic)

            # Store the results
            search_results['path'] = path
            search_results['visited'] = visited
            search_results['metrics'] = metrics
            search_results['states'] = states

            # Update the slider maximum
            step_slider.max = len(states['steps']) - 1
            step_slider.value = 0

            # Display results
            if path:
                print(
                    f"✓ Path found! Length: {len(path) - 1}, Cost: {metrics['path_cost']}")
                print(f"Path: {' -> '.join(path)}")
            else:
                print("✗ No path found!")

            print(f"Nodes visited: {metrics['nodes_visited']}")
            print(f"Max frontier size: {metrics['space']}")
            print(f"Time taken: {metrics['time']:.6f} seconds")
            print(
                "\nUse the slider or Next Step button to step through the search process.")

            # Update the graph
            update_graph(0)

    # Function for next step button
    def on_next_step_clicked(b):
        if not search_results or 'states' not in search_results:
            with output:
                clear_output()
                print("Run search first before stepping through.")
            return

        states = search_results.get('states')
        if not states or not states['steps']:
            return

        current_step = step_slider.value
        if current_step < len(states['steps']) - 1:
            step_slider.value += 1

    # Connect events
    run_button.on_click(on_run_button_clicked)
    step_slider.observe(on_step_change, names='value')
    next_step_button.on_click(on_next_step_clicked)

    # Initial graph display
    with graph_output:
        fig = visualize_graph(
            demo_graph, title="Demo Graph (Run search to see steps)")
        plt.show()

    # Layout the widgets
    top_row = widgets.HBox(
        [algorithm_dropdown, start_node_dropdown, goal_node_dropdown, heuristic_dropdown])
    middle_row = widgets.HBox([run_button, step_slider, next_step_button])

    # Display everything
    display(widgets.VBox([
        top_row,
        middle_row,
        output,
        graph_output  # Contains both graph and metadata now
    ]))

if __name__ == "__main__":

    romania_map = create_romania_map()
    run_graph_search_example(romania_map, 'bfs', 'Arad', 'Bucharest')
    run_graph_search_example(romania_map, 'astar', 'Arad', 'Bucharest', 'euclidean')

    # demo_graph = create_demo_graph()
    # run_graph_search_example(demo_graph, 'astar', 'A', 'M', 'manhattan')