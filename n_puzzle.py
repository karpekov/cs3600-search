import time
import heapq
import random
import copy
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import ipywidgets as widgets
from IPython.display import display, clear_output

class NPuzzle:
    """
    N-Puzzle implementation that can represent both 8-puzzle (3x3) and 15-puzzle (4x4).
    """
    def __init__(self, size=3, state=None):
        """
        Initialize an N-puzzle with a given size (3 for 8-puzzle, 4 for 15-puzzle).
        If state is provided, use it as the initial state, otherwise generate a random solvable state.
        """
        self.size = size
        self.goal_state = tuple(range(size * size))  # 0 represents the empty space

        if state is not None:
            self.state = state
        else:
            # Generate a random solvable state
            self.state = self.generate_solvable_state()

    def generate_solvable_state(self):
        """
        Generate a random solvable state for the N-puzzle.
        """
        while True:
            # Generate a random permutation
            state = list(range(self.size * self.size))
            random.shuffle(state)
            state = tuple(state)

            # Check if it's solvable
            if self.is_solvable(state):
                return state

    def is_solvable(self, state):
        """
        Check if the given state is solvable.
        For an N-puzzle, a state is solvable if:
        - For odd N: the number of inversions is even
        - For even N: the number of inversions + row of blank from bottom is even
        """
        # Count inversions
        inversions = 0
        state_list = list(state)

        for i in range(len(state_list)):
            if state_list[i] == 0:  # Skip the empty tile
                continue
            for j in range(i + 1, len(state_list)):
                if state_list[j] == 0:  # Skip the empty tile
                    continue
                if state_list[i] > state_list[j]:
                    inversions += 1

        # Find the position of the blank tile
        blank_idx = state_list.index(0)
        blank_row = blank_idx // self.size
        blank_row_from_bottom = self.size - blank_row - 1

        # Check solvability based on the size of the puzzle
        if self.size % 2 == 1:  # Odd size (e.g., 3x3)
            return inversions % 2 == 0
        else:  # Even size (e.g., 4x4)
            return (inversions + blank_row_from_bottom) % 2 == 0

    def get_blank_position(self, state=None):
        """
        Find the position of the blank (0) in the state.
        """
        if state is None:
            state = self.state
        return state.index(0)

    def get_possible_moves(self, state=None):
        """
        Get all possible moves from the current state.
        """
        if state is None:
            state = self.state

        state_list = list(state)
        blank_idx = state_list.index(0)
        blank_row = blank_idx // self.size
        blank_col = blank_idx % self.size

        possible_moves = []

        # Check all four directions: up, right, down, left
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        direction_names = ['up', 'right', 'down', 'left']

        for (dr, dc), direction in zip(directions, direction_names):
            new_row = blank_row + dr
            new_col = blank_col + dc

            # Check if the move is valid
            if 0 <= new_row < self.size and 0 <= new_col < self.size:
                new_idx = new_row * self.size + new_col

                # Create a new state by swapping the blank with the adjacent tile
                new_state = list(state)
                new_state[blank_idx], new_state[new_idx] = new_state[new_idx], new_state[blank_idx]

                possible_moves.append((tuple(new_state), direction))

        return possible_moves

    def is_goal(self, state=None):
        """
        Check if the given state is the goal state.
        """
        if state is None:
            state = self.state
        return state == self.goal_state

    def visualize(self, state=None, title="N-Puzzle"):
        """
        Visualize the N-puzzle state.
        """
        if state is None:
            state = self.state

        fig = plt.figure(figsize=(5, 5))
        plt.title(title)

        # Create a grid
        for i in range(self.size + 1):
            plt.axhline(y=i, color='black', linestyle='-')
            plt.axvline(x=i, color='black', linestyle='-')

        # Fill in the numbers
        for i in range(self.size):
            for j in range(self.size):
                idx = i * self.size + j
                value = state[idx]
                if value != 0:  # Skip the blank tile
                    plt.text(j + 0.5, i + 0.5, str(value), fontsize=20, ha='center', va='center')

        plt.xlim(0, self.size)
        plt.ylim(self.size, 0)  # Invert y-axis to match puzzle orientation
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        return fig

    def manhattan_distance(self, state=None):
        """
        Calculate the Manhattan distance heuristic for the given state.
        """
        if state is None:
            state = self.state

        distance = 0
        for i in range(len(state)):
            if state[i] == 0:  # Skip the blank tile
                continue

            # Calculate the expected position of the tile
            expected_row = state[i] // self.size
            expected_col = state[i] % self.size

            # Calculate the current position of the tile
            current_row = i // self.size
            current_col = i % self.size

            # Add the Manhattan distance
            distance += abs(expected_row - current_row) + abs(expected_col - current_col)

        return distance

    def misplaced_tiles(self, state=None):
        """
        Calculate the number of misplaced tiles heuristic for the given state.
        """
        if state is None:
            state = self.state

        count = 0
        for i in range(len(state)):
            if state[i] != 0 and state[i] != i:  # Skip the blank tile and correctly placed tiles
                count += 1

        return count

    def linear_conflict(self, state=None):
        """
        Calculate the linear conflict heuristic for the given state.
        This is the Manhattan distance plus a penalty for linear conflicts.
        """
        if state is None:
            state = self.state

        # Start with the Manhattan distance
        distance = self.manhattan_distance(state)
        conflicts = 0

        # Check for linear conflicts in rows
        for row in range(self.size):
            for i in range(self.size):
                idx1 = row * self.size + i
                tile1 = state[idx1]

                if tile1 == 0 or tile1 // self.size != row:
                    continue  # Skip the blank tile and tiles that don't belong in this row

                for j in range(i + 1, self.size):
                    idx2 = row * self.size + j
                    tile2 = state[idx2]

                    if tile2 == 0 or tile2 // self.size != row:
                        continue  # Skip the blank tile and tiles that don't belong in this row

                    if tile1 > tile2:  # If they're in the wrong order
                        conflicts += 1

        # Check for linear conflicts in columns
        for col in range(self.size):
            for i in range(self.size):
                idx1 = i * self.size + col
                tile1 = state[idx1]

                if tile1 == 0 or tile1 % self.size != col:
                    continue  # Skip the blank tile and tiles that don't belong in this column

                for j in range(i + 1, self.size):
                    idx2 = j * self.size + col
                    tile2 = state[idx2]

                    if tile2 == 0 or tile2 % self.size != col:
                        continue  # Skip the blank tile and tiles that don't belong in this column

                    if tile1 > tile2:  # If they're in the wrong order
                        conflicts += 1

        # Each conflict requires at least two moves to resolve
        return distance + 2 * conflicts

def n_puzzle_search(puzzle, algorithm='bfs', heuristic_name=None, max_iterations=10000):
    """
    Solve the N-puzzle using the specified search algorithm.

    Args:
        puzzle: NPuzzle instance
        algorithm: Search algorithm to use ('bfs', 'dfs', 'ucs', 'greedy', 'astar')
        heuristic_name: Name of the heuristic to use ('manhattan', 'misplaced', 'linear_conflict')
        max_iterations: Maximum number of iterations to prevent infinite loops

    Returns:
        path: List of states in the solution path
        actions: List of actions to reach the goal
        metrics: Dictionary with metrics (time, space, path_cost)
        states: Dictionary with states for visualization
    """
    start_time = time.time()
    start_state = puzzle.state

    # Select heuristic function based on name
    heuristic_func = None
    if heuristic_name == 'manhattan':
        heuristic_func = puzzle.manhattan_distance
    elif heuristic_name == 'misplaced':
        heuristic_func = puzzle.misplaced_tiles
    elif heuristic_name == 'linear_conflict':
        heuristic_func = puzzle.linear_conflict

    # Initialize the frontier based on the algorithm
    if algorithm == 'bfs':
        frontier = deque([(start_state, [], 0, [start_state])])  # (state, actions, cost, path)
    elif algorithm == 'dfs':
        frontier = [(start_state, [], 0, [start_state])]  # Using list as stack
    elif algorithm in ['ucs', 'greedy', 'astar']:
        # For priority queue: (priority, state_id, state, actions, cost, path)
        # We need state_id to break ties and avoid comparing states directly
        if algorithm == 'ucs':
            frontier = [(0, id(start_state), start_state, [], 0, [start_state])]  # Priority is cost
        elif algorithm == 'greedy':
            if not heuristic_func:
                raise ValueError("Heuristic function required for greedy search")
            h = heuristic_func(start_state)
            frontier = [(h, id(start_state), start_state, [], 0, [start_state])]  # Priority is heuristic
        elif algorithm == 'astar':
            if not heuristic_func:
                raise ValueError("Heuristic function required for A* search")
            h = heuristic_func(start_state)
            frontier = [(h, id(start_state), start_state, [], 0, [start_state])]  # Priority is f(n) = g(n) + h(n)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # Use a set to keep track of visited states
    visited = set([start_state])
    max_frontier_size = 1
    iterations = 0

    # For visualization
    states = {
        'steps': []
    }

    while frontier and iterations < max_iterations:
        iterations += 1
        max_frontier_size = max(max_frontier_size, len(frontier))

        # Get the next state based on the algorithm
        if algorithm == 'bfs':
            current_state, actions, cost, path = frontier.popleft()
        elif algorithm == 'dfs':
            current_state, actions, cost, path = frontier.pop()
        elif algorithm in ['ucs', 'greedy', 'astar']:
            priority, _, current_state, actions, cost, path = heapq.heappop(frontier)

        # Save state for visualization (every 10 iterations to reduce memory usage)
        if iterations % 10 == 0 or iterations < 10:
            h_value = None
            if heuristic_func:
                h_value = heuristic_func(current_state)

            # Get only valid moves from the current state
            valid_moves = puzzle.get_possible_moves(current_state)
            frontier_nodes = [move[0] for move in valid_moves]

            states['steps'].append({
                'state': current_state,
                'current': current_state,
                'path': path.copy() if path else [],
                'visited': list(visited),
                'frontier': frontier_nodes,
                'actions': actions.copy() if actions else [],
                'cost': cost,
                'heuristic': h_value,
                'frontier_size': len(frontier),
                'visited_size': len(visited),
                'iteration': iterations
            })

        # Check if we've reached the goal
        if puzzle.is_goal(current_state):
            end_time = time.time()
            metrics = {
                'time': end_time - start_time,
                'space': max_frontier_size,
                'path_cost': cost,
                'iterations': iterations,
                'states_visited': len(visited)
            }

            # Use the current state as the end of path
            return path, actions, metrics, states

        # Expand the current state
        for next_state, action in puzzle.get_possible_moves(current_state):
            if next_state not in visited:
                visited.add(next_state)
                new_actions = actions + [action]
                new_cost = cost + 1  # Each move has a cost of 1
                new_path = path + [next_state]

                if algorithm == 'bfs':
                    frontier.append((next_state, new_actions, new_cost, new_path))
                elif algorithm == 'dfs':
                    frontier.append((next_state, new_actions, new_cost, new_path))
                elif algorithm == 'ucs':
                    heapq.heappush(frontier, (new_cost, id(next_state), next_state, new_actions, new_cost, new_path))
                elif algorithm == 'greedy':
                    h = heuristic_func(next_state)
                    heapq.heappush(frontier, (h, id(next_state), next_state, new_actions, new_cost, new_path))
                elif algorithm == 'astar':
                    h = heuristic_func(next_state)
                    f = new_cost + h  # f(n) = g(n) + h(n)
                    heapq.heappush(frontier, (f, id(next_state), next_state, new_actions, new_cost, new_path))

    # If we reach here, no solution was found within the iteration limit
    end_time = time.time()
    metrics = {
        'time': end_time - start_time,
        'space': max_frontier_size,
        'path_cost': float('inf'),
        'iterations': iterations,
        'states_visited': len(visited)
    }
    return [], [], metrics, states

def create_specific_puzzle(size, difficulty):
    """
    Create a puzzle with a specific difficulty level.
    Difficulty is determined by the number of random moves from the goal state.
    """
    puzzle = NPuzzle(size=size, state=tuple(range(size * size)))
    state = puzzle.goal_state

    # Apply random moves to get a solvable state with the desired difficulty
    moves = 0
    visited_states = set([state])

    while moves < difficulty:
        possible_moves = puzzle.get_possible_moves(state)
        if not possible_moves:
            break

        next_state, _ = random.choice(possible_moves)

        if next_state not in visited_states:
            state = next_state
            visited_states.add(state)
            moves += 1

    return NPuzzle(size=size, state=state)

def run_search_comparison_puzzle(puzzle, algorithms, heuristics=None, max_iterations=10000):
    """
    Run multiple search algorithms on the same puzzle and collect metrics.

    Args:
        puzzle: NPuzzle instance
        algorithms: List of algorithm names
        heuristics: Dictionary mapping algorithm names to heuristic names
        max_iterations: Maximum iterations for each search

    Returns:
        results: Dictionary with search results and metrics
    """
    results = {}

    for algorithm in algorithms:
        heuristic_name = None
        if algorithm in ['greedy', 'astar'] and heuristics:
            heuristic_name = heuristics.get(algorithm)

        path, actions, metrics, states = n_puzzle_search(
            puzzle, algorithm, heuristic_name, max_iterations
        )

        results[algorithm] = {
            'path': path,
            'actions': actions,
            'metrics': metrics,
            'states': states
        }

    return results


def create_n_puzzle_demo(
        size=3,
        difficulty=5,
):
    demo_puzzle = create_specific_puzzle(size=3, difficulty=5)

    # Function to run a search on an N-puzzle and collect step information
    def run_n_puzzle_demo(algorithm, puzzle, heuristic_name, max_iterations=1000):
        path, actions, metrics, states = n_puzzle_search(
            puzzle, algorithm, heuristic_name, max_iterations
        )
        return path, actions, metrics, states

    # Create interactive widgets
    algorithm_dropdown_puzzle = widgets.Dropdown(
        options=['BFS', 'DFS', 'UCS', 'Greedy', 'Astar'],
        value='Astar',
        description='Algorithm:',
        style={'description_width': 'initial'}
    )

    puzzle_size_dropdown = widgets.Dropdown(
        options=[("8-Puzzle (3x3)", 3), ("15-Puzzle (4x4)", 4)],
        value=3,
        description='Puzzle Size:',
        style={'description_width': 'initial'}
    )

    difficulty_slider = widgets.IntSlider(
        value=5,
        min=1,
        max=200,
        step=1,
        description='Difficulty:',
        style={'description_width': 'initial'}
    )

    heuristic_dropdown_puzzle = widgets.Dropdown(
        options=['manhattan', 'misplaced', 'linear_conflict'],
        value='manhattan',
        description='Heuristic:',
        style={'description_width': 'initial'}
    )

    step_slider_puzzle = widgets.IntSlider(
        min=0,
        max=1,
        step=1,
        value=0,
        description='Step:',
        style={'description_width': 'initial'},
        continuous_update=False
    )

    max_iterations_slider = widgets.IntSlider(
        value=1000,
        min=100,
        max=10000,
        step=100,
        description='Max Iterations:',
        style={'description_width': 'initial'}
    )

    create_puzzle_button = widgets.Button(
        description='Create Puzzle',
        button_style='info',
        icon='refresh'
    )

    run_button_puzzle = widgets.Button(
        description='Run Search',
        button_style='success',
        icon='play'
    )

    # Add Next Step button
    next_step_button = widgets.Button(
        description='Next Step',
        button_style='info',
        icon='arrow-right'
    )

    output_puzzle = widgets.Output()
    puzzle_output = widgets.Output()

    # Store puzzle and search results
    puzzle_state = {
        'puzzle': demo_puzzle,
        'results': {}
    }

    def update_puzzle_display(step):
        with puzzle_output:
            clear_output(wait=True)

            if not puzzle_state['results']:
                # Just show the current puzzle state
                fig = puzzle_state['puzzle'].visualize(
                    title="N-Puzzle (Run search to see steps)")
                plt.show()
                return

            states = puzzle_state['results'].get('states')
            if not states or not states['steps'] or step >= len(states['steps']):
                fig = puzzle_state['puzzle'].visualize(
                    title="N-Puzzle (No steps available)")
                plt.show()
                return

            # Get the current state information
            current_state = states['steps'][step]
            puzzle_state_to_show = current_state['state']
            iteration = current_state['iteration']
            cost = current_state['cost']
            h_value = current_state['heuristic']

            # Determine the title based on the algorithm
            if algorithm_dropdown_puzzle.value.lower() in ['greedy', 'astar']:
                title = f"Step {iteration}: {algorithm_dropdown_puzzle.value} Search - g={cost}, h={h_value}"
            else:
                title = f"Step {iteration}: {algorithm_dropdown_puzzle.value} Search - Cost={cost}"

            # Display the puzzle state
            fig = puzzle_state['puzzle'].visualize(
                puzzle_state_to_show, title=title)
            plt.show()

            # Display metadata below the puzzle instead of in a separate output
            if not puzzle_state['results']:
                print("Run search first to see step information.")
                return

            if not states or not states['steps'] or step >= len(states['steps']):
                print("No step information available.")
                return

            # Get the current state information
            actions = current_state['actions']
            frontier_size = current_state['frontier_size']
            visited_size = current_state['visited_size']

            print(f"Iteration: {iteration}")
            print(
                f"Actions so far: {', '.join(actions) if actions else 'None'}")
            print(f"Cost (g): {cost}")

            if h_value is not None:
                print(f"Heuristic value (h): {h_value}")
                if algorithm_dropdown_puzzle.value.lower() == 'astar':
                    print(f"f(n) = g(n) + h(n) = {cost + h_value}")

            print(f"Frontier size: {frontier_size}")
            print(f"Visited states: {visited_size}")

            # If we're at the goal state (last step in the path)
            path = puzzle_state['results'].get('path')
            if path and puzzle_state['puzzle'].is_goal(current_state['state']):
                print("\n🎯 Goal state reached!")
                metrics = puzzle_state['results'].get('metrics')
                if metrics:
                    print(f"\nSearch Metrics:")
                    print(f"Time taken: {metrics['time']:.6f} seconds")
                    print(f"Max frontier size: {metrics['space']}")
                    print(f"Total states visited: {metrics['states_visited']}")
                    print(f"Total iterations: {metrics['iterations']}")

    def on_step_puzzle_change(change):
        if change['name'] == 'value':
            update_puzzle_display(change['new'])

    def on_create_puzzle_clicked(b):
        with output_puzzle:
            clear_output()
            size = puzzle_size_dropdown.value
            difficulty = difficulty_slider.value

            print(
                f"Creating {size}x{size} puzzle with difficulty {difficulty}...")

            # Create a new puzzle
            new_puzzle = create_specific_puzzle(size, difficulty)
            puzzle_state['puzzle'] = new_puzzle
            puzzle_state['results'] = {}

            print(f"✓ New puzzle created!")

            # Reset step slider
            step_slider_puzzle.max = 0
            step_slider_puzzle.value = 0

            # Update display
            with puzzle_output:
                clear_output(wait=True)
                fig = new_puzzle.visualize(
                    title=f"{size}x{size} Puzzle - Difficulty {difficulty}")
                plt.show()
                print(
                    f"{size}x{size} puzzle created with difficulty level {difficulty}.")
                print("Click 'Run Search' to solve the puzzle.")

    def on_run_puzzle_button_clicked(b):
        with output_puzzle:
            clear_output()
            algorithm = algorithm_dropdown_puzzle.value.lower()
            heuristic = heuristic_dropdown_puzzle.value
            max_iterations = max_iterations_slider.value

            puzzle = puzzle_state['puzzle']
            size = puzzle.size

            # Check if we need a heuristic for the algorithm
            if algorithm in ['greedy', 'astar'] and not heuristic:
                print("⚠️ Error: Greedy and A* search require a heuristic.")
                return

            print(f"Running {algorithm} search on {size}x{size} puzzle")
            if algorithm in ['greedy', 'astar']:
                print(f"Using {heuristic} heuristic")
            print(f"Maximum iterations: {max_iterations}")

            # Run the search
            path, actions, metrics, states = run_n_puzzle_demo(
                algorithm, puzzle, heuristic, max_iterations
            )

            # Store the results
            puzzle_state['results'] = {
                'path': path,
                'actions': actions,
                'metrics': metrics,
                'states': states
            }

            # Update the slider maximum
            step_slider_puzzle.max = len(states['steps']) - 1
            step_slider_puzzle.value = 0

            # Display results
            if path:
                print(f"✓ Solution found!")
                print(f"Path length: {len(path) - 1} moves")
                print(f"Actions: {', '.join(actions)}")
            else:
                print("✗ No solution found within iteration limit!")

            print(f"States visited: {metrics['states_visited']}")
            print(f"Max frontier size: {metrics['space']}")
            print(f"Iterations performed: {metrics['iterations']}")
            print(f"Time taken: {metrics['time']:.6f} seconds")
            print(
                "\nUse the slider or Next Step button to step through the search process.")

            # Update the display
            update_puzzle_display(0)

    # Function for next step button
    def on_next_step_clicked(b):
        if not puzzle_state['results'] or 'states' not in puzzle_state['results']:
            with output_puzzle:
                clear_output()
                print("Run search first before stepping through.")
            return

        states = puzzle_state['results'].get('states')
        if not states or not states['steps']:
            return

        current_step = step_slider_puzzle.value
        if current_step < len(states['steps']) - 1:
            step_slider_puzzle.value += 1

    # Connect events
    create_puzzle_button.on_click(on_create_puzzle_clicked)
    run_button_puzzle.on_click(on_run_puzzle_button_clicked)
    step_slider_puzzle.observe(on_step_puzzle_change, names='value')
    next_step_button.on_click(on_next_step_clicked)

    # Initial puzzle display
    with puzzle_output:
        fig = demo_puzzle.visualize(title="8-Puzzle (Run search to see steps)")
        plt.show()
        print("Click 'Run Search' to solve the puzzle.")

    # Layout the widgets
    top_row_puzzle = widgets.HBox(
        [algorithm_dropdown_puzzle, puzzle_size_dropdown, difficulty_slider])
    middle_row_puzzle = widgets.HBox(
        [heuristic_dropdown_puzzle, max_iterations_slider])
    button_row_puzzle = widgets.HBox(
        [create_puzzle_button, run_button_puzzle, step_slider_puzzle, next_step_button])

    # Display everything
    display(widgets.VBox([
        top_row_puzzle,
        middle_row_puzzle,
        button_row_puzzle,
        output_puzzle,
        puzzle_output  # Now contains both visualization and metadata
    ]))

