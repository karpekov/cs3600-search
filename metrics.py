import time
import random
import pandas as pd
import numpy as np
import altair as alt
from collections import defaultdict

from graph_search import create_demo_graph, create_large_demo_graph, graph_search, euclidean_distance_heuristic, manhattan_distance_heuristic
from n_puzzle import create_specific_puzzle, n_puzzle_search

def run_graph_search_simulations(num_simulations=50, algorithms=None, start_nodes=None, goal_nodes=None):
    """
    Run multiple simulations of graph search algorithms and collect metrics.

    Args:
        num_simulations: Number of simulations to run
        algorithms: List of algorithms to run (default: all)
        start_nodes: List of start nodes (default: random selection)
        goal_nodes: List of goal nodes (default: random selection)

    Returns:
        DataFrame with simulation results
    """
    if algorithms is None:
        algorithms = ['bfs', 'dfs', 'ucs', 'greedy', 'astar']

    # Create a demo graph
    G = create_large_demo_graph()
    nodes = list(G.nodes())

    # Prepare heuristic functions
    heuristics = {
        'euclidean': lambda node1, node2: euclidean_distance_heuristic(node1, node2, G),
        'manhattan': lambda node1, node2: manhattan_distance_heuristic(node1, node2, G)
    }

    # If start_nodes and goal_nodes are not provided, create random pairs
    if start_nodes is None or goal_nodes is None:
        pairs = []
        for _ in range(num_simulations):
            # Select random start and goal nodes
            start = random.choice(nodes)
            goal = random.choice([n for n in nodes if n != start])
            pairs.append((start, goal))
    else:
        # Use provided start and goal nodes, and repeat if necessary
        pairs = []
        for i in range(num_simulations):
            idx = i % len(start_nodes)
            pairs.append((start_nodes[idx], goal_nodes[idx]))

    # Run simulations and collect results
    results = []

    for i, (start, goal) in enumerate(pairs):
        for algorithm in algorithms:
            heuristic = None
            heuristic_name = "None"

            if algorithm in ['greedy', 'astar']:
                for h_name, h_func in heuristics.items():
                    # Use the function as heuristic
                    path, visited, metrics, _ = graph_search(G, start, goal, algorithm, h_func)

                    results.append({
                        'simulation': i,
                        'algorithm': f"{algorithm}_{h_name}",
                        'start': start,
                        'goal': goal,
                        'path_found': path is not None,
                        'path_length': len(path) - 1 if path else float('inf'),
                        'path_cost': metrics['path_cost'],
                        'nodes_visited': metrics['nodes_visited'],
                        'max_frontier_size': metrics['space'],
                        'time': metrics['time']
                    })
            else:
                # Run uninformed search
                path, visited, metrics, _ = graph_search(G, start, goal, algorithm, None)

                results.append({
                    'simulation': i,
                    'algorithm': algorithm,
                    'start': start,
                    'goal': goal,
                    'path_found': path is not None,
                    'path_length': len(path) - 1 if path else float('inf'),
                    'path_cost': metrics['path_cost'],
                    'nodes_visited': metrics['nodes_visited'],
                    'max_frontier_size': metrics['space'],
                    'time': metrics['time']
                })

    return pd.DataFrame(results)

def run_n_puzzle_simulations(num_simulations=30, algorithms=None, puzzle_sizes=None, difficulties=None):
    """
    Run multiple simulations of N-puzzle search algorithms and collect metrics.

    Args:
        num_simulations: Number of simulations to run
        algorithms: List of algorithms to run (default: all)
        puzzle_sizes: List of puzzle sizes (default: [3])
        difficulties: List of difficulties (default: [5, 10, 15])

    Returns:
        DataFrame with simulation results
    """
    if algorithms is None:
        algorithms = ['bfs', 'dfs', 'ucs', 'greedy', 'astar']

    if puzzle_sizes is None:
        puzzle_sizes = [3]  # Default to 8-puzzle (3x3)

    if difficulties is None:
        difficulties = [5, 10, 15]  # Default difficulties

    # Prepare heuristic names
    heuristic_names = ['manhattan', 'misplaced', 'linear_conflict']

    # Run simulations and collect results
    results = []

    for i in range(num_simulations):
        for size in puzzle_sizes:
            for difficulty in difficulties:
                # Create a puzzle with specified size and difficulty
                puzzle = create_specific_puzzle(size, difficulty)

                for algorithm in algorithms:
                    if algorithm in ['greedy', 'astar']:
                        for h_name in heuristic_names:
                            try:
                                path, actions, metrics, _ = n_puzzle_search(
                                    puzzle, algorithm, h_name, max_iterations=5000
                                )

                                results.append({
                                    'simulation': i,
                                    'algorithm': f"{algorithm}_{h_name}",
                                    'puzzle_size': size,
                                    'difficulty': difficulty,
                                    'path_found': path is not None,
                                    'path_length': len(path) - 1 if path else float('inf'),
                                    'path_cost': metrics['path_cost'],
                                    'states_visited': metrics['states_visited'],
                                    'max_frontier_size': metrics['space'],
                                    'iterations': metrics['iterations'],
                                    'time': metrics['time']
                                })
                            except Exception as e:
                                # Skip if the search fails
                                print(f"Error in simulation {i}, algorithm {algorithm}_{h_name}: {e}")
                    else:
                        try:
                            # Run uninformed search (with a lower iteration limit for DFS)
                            max_iter = 500 if algorithm == 'dfs' else 5000
                            path, actions, metrics, _ = n_puzzle_search(
                                puzzle, algorithm, None, max_iterations=max_iter
                            )

                            results.append({
                                'simulation': i,
                                'algorithm': algorithm,
                                'puzzle_size': size,
                                'difficulty': difficulty,
                                'path_found': path is not None,
                                'path_length': len(path) - 1 if path else float('inf'),
                                'path_cost': metrics['path_cost'],
                                'states_visited': metrics['states_visited'],
                                'max_frontier_size': metrics['space'],
                                'iterations': metrics['iterations'],
                                'time': metrics['time']
                            })
                        except Exception as e:
                            # Skip if the search fails
                            print(f"Error in simulation {i}, algorithm {algorithm}: {e}")

    return pd.DataFrame(results)

def create_performance_charts(df, metric='time', title=None):
    """
    Create Altair charts for performance metrics.

    Args:
        df: DataFrame with simulation results
        metric: Metric to visualize ('time', 'nodes_visited', 'max_frontier_size', 'path_length', 'path_cost')
        title: Chart title

    Returns:
        Altair chart
    """
    if title is None:
        title = f"Performance Comparison: {metric.replace('_', ' ').title()}"

    # Calculate summary statistics for each algorithm
    summary = df.groupby('algorithm')[metric].agg(['mean', 'median', 'std', 'min', 'max']).reset_index()

    # Create a bar chart with error bars
    bars = alt.Chart(summary).mark_bar().encode(
        x=alt.X('algorithm:N', title='Algorithm', sort=None),
        y=alt.Y('mean:Q', title=metric.replace('_', ' ').title()),
        color=alt.Color('algorithm:N', legend=None)
    )

    # Add error bars
    error_bars = alt.Chart(summary).mark_errorbar().encode(
        x='algorithm:N',
        y='min:Q',
        y2='max:Q'
    )

    # Combine the charts
    chart = (bars + error_bars).properties(
        title=title,
        width=600,
        height=400
    )

    return chart

def create_heatmap(df, value_col='time', title=None):
    """
    Create a heatmap to compare algorithm performance across different conditions.

    Args:
        df: DataFrame with simulation results
        value_col: Column to use for heatmap values
        title: Chart title

    Returns:
        Altair chart
    """
    if 'difficulty' in df.columns:
        # For N-puzzle, compare across difficulties
        if title is None:
            title = f"{value_col.replace('_', ' ').title()} vs Difficulty by Algorithm"

        # Calculate mean values for each algorithm and difficulty
        summary = df.groupby(['algorithm', 'difficulty'])[value_col].mean().reset_index()

        # Create a heatmap
        heatmap = alt.Chart(summary).mark_rect().encode(
            x=alt.X('difficulty:O', title='Difficulty'),
            y=alt.Y('algorithm:N', title='Algorithm', sort=None),
            color=alt.Color(f'{value_col}:Q', title=value_col.replace('_', ' ').title(),
                          scale=alt.Scale(scheme='viridis')),
            tooltip=['algorithm', 'difficulty', alt.Tooltip(f'{value_col}:Q', format='.2f')]
        ).properties(
            title=title,
            width=500,
            height=alt.Step(40)
        )

        return heatmap

    else:
        # For graph search, compare start-goal pairs
        if title is None:
            title = f"{value_col.replace('_', ' ').title()} by Algorithm and Path"

        # Create a unique path identifier
        df['path'] = df['start'] + '->' + df['goal']

        # Calculate mean values for each algorithm and path
        summary = df.groupby(['algorithm', 'path'])[value_col].mean().reset_index()

        # Create a heatmap
        heatmap = alt.Chart(summary).mark_rect().encode(
            x=alt.X('path:N', title='Start -> Goal'),
            y=alt.Y('algorithm:N', title='Algorithm', sort=None),
            color=alt.Color(f'{value_col}:Q', title=value_col.replace('_', ' ').title(),
                          scale=alt.Scale(scheme='viridis')),
            tooltip=['algorithm', 'path', alt.Tooltip(f'{value_col}:Q', format='.2f')]
        ).properties(
            title=title,
            width=500,
            height=alt.Step(40)
        )

        return heatmap

def create_boxplot(df, value_col='time', title=None):
    """
    Create a boxplot to show the distribution of a metric across algorithms.

    Args:
        df: DataFrame with simulation results
        value_col: Column to use for boxplot values
        title: Chart title

    Returns:
        Altair chart
    """
    if title is None:
        title = f"Distribution of {value_col.replace('_', ' ').title()} by Algorithm"

    # Create a boxplot
    boxplot = alt.Chart(df).mark_boxplot().encode(
        x=alt.X('algorithm:N', title='Algorithm'),
        y=alt.Y(f'{value_col}:Q', title=value_col.replace('_', ' ').title()),
        color='algorithm:N'
    ).properties(
        title=title,
        width=600,
        height=400
    )

    return boxplot

def create_path_distribution_chart(df, title=None):
    """
    Create a chart showing the distribution of path lengths for different algorithms.

    Args:
        df: DataFrame with simulation results
        title: Chart title

    Returns:
        Altair chart
    """
    if title is None:
        title = "Path Length Distribution by Algorithm"

    # Filter out cases where no path was found
    filtered_df = df[df['path_found'] == True].copy()

    # Create a chart for path length distribution
    chart = alt.Chart(filtered_df).mark_bar().encode(
        x=alt.X('path_length:Q', bin=alt.Bin(maxbins=20), title='Path Length'),
        y=alt.Y('count()', title='Count'),
        color='algorithm:N',
        column='algorithm:N'
    ).properties(
        title=title,
        width=150,
        height=300
    )

    return chart

def create_success_rate_chart(df, title=None):
    """
    Create a chart showing the success rate of different algorithms.

    Args:
        df: DataFrame with simulation results
        title: Chart title

    Returns:
        Altair chart
    """
    if title is None:
        title = "Success Rate by Algorithm"

    # Calculate success rate for each algorithm
    success_df = df.groupby('algorithm')['path_found'].mean().reset_index()
    success_df['success_rate'] = success_df['path_found'] * 100

    # Create a bar chart for success rate
    chart = alt.Chart(success_df).mark_bar().encode(
        x=alt.X('algorithm:N', title='Algorithm'),
        y=alt.Y('success_rate:Q', title='Success Rate (%)'),
        color='algorithm:N',
        tooltip=['algorithm', alt.Tooltip('success_rate:Q', format='.1f')]
    ).properties(
        title=title,
        width=500,
        height=300
    )

    return chart