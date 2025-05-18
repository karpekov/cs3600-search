import time
import random
import pandas as pd
import numpy as np
import altair as alt
from collections import defaultdict
import os

from graph_search import create_demo_graph, create_large_demo_graph, graph_search, euclidean_distance_heuristic, manhattan_distance_heuristic
from n_puzzle import create_specific_puzzle, n_puzzle_search
# Add import for word ladder
try:
    from word_ladder import WordLadderGame, hamming_distance, letter_set_difference, vowel_consonant_difference
except ImportError:
    print("Word ladder module not found. Word ladder metrics will not be available.")

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

def run_word_ladder_simulations(num_simulations=30, algorithms=None, word_lengths=None, use_nltk=True):
    """
    Run multiple simulations of word ladder search algorithms and collect metrics.

    Args:
        num_simulations: Number of simulations to run per word length
        algorithms: List of algorithms to run (default: all)
        word_lengths: List of word lengths to test (default: [3, 4, 5])
        use_nltk: Whether to use NLTK for word corpus

    Returns:
        DataFrame with simulation results
    """
    if 'WordLadderGame' not in globals():
        print("Word ladder module not imported. Cannot run simulations.")
        return pd.DataFrame()

    if algorithms is None:
        algorithms = ['bfs', 'dfs', 'ucs', 'greedy', 'astar']

    if word_lengths is None:
        word_lengths = [3, 4, 5]

    # Prepare heuristic names and functions
    heuristics = {
        'hamming': hamming_distance,
        'letter_set': letter_set_difference,
        'vowel_consonant': vowel_consonant_difference
    }

    # Ensure word list files exist
    WordLadderGame.generate_word_list_files()

    # Run simulations and collect results
    results = []

    # Words for testing that are known to have paths
    test_word_pairs = {
        3: [('cat', 'dog'), ('hit', 'cog'), ('hot', 'dog'), ('bat', 'rat')],
        4: [('play', 'work'), ('cold', 'warm'), ('word', 'talk'), ('make', 'take')],
        5: [('light', 'night'), ('world', 'peace'), ('house', 'place'), ('sound', 'water')]
    }

    for word_length in word_lengths:
        # Create a game with the specified word length
        try:
            game = WordLadderGame(word_length=word_length, use_nltk=use_nltk)
            print(f"Created game with {len(game.words)} {word_length}-letter words")

            if len(game.words) < 2:
                print(f"Not enough {word_length}-letter words, skipping")
                continue

            # Get pairs to test
            word_pairs = []

            # First add any known test pairs that exist in our dictionary
            for start, target in test_word_pairs.get(word_length, []):
                if start in game.words and target in game.words:
                    word_pairs.append((start, target))

            # If we don't have enough pairs, add some random ones
            words_list = list(game.words)
            while len(word_pairs) < num_simulations:
                # Select random start and target words
                start = random.choice(words_list)
                # Try to find a word that is a reasonable distance from start
                candidates = [w for w in words_list if hamming_distance(start, w) > 1 and w != start]
                if candidates:
                    target = random.choice(candidates)
                    word_pairs.append((start, target))
                else:
                    # Just pick any different word
                    target = random.choice([w for w in words_list if w != start])
                    word_pairs.append((start, target))

            # Run the simulations
            for i, (start, target) in enumerate(word_pairs):
                print(f"Running simulation {i+1}/{len(word_pairs)} for {word_length}-letter words: {start} -> {target}")

                for algorithm in algorithms:
                    if algorithm in ['greedy', 'astar']:
                        for h_name, h_func in heuristics.items():
                            try:
                                path, visited, metrics, _ = game.find_path(
                                    start, target, algorithm=algorithm, heuristic=h_func
                                )

                                results.append({
                                    'simulation': i,
                                    'algorithm': f"{algorithm}_{h_name}",
                                    'word_length': word_length,
                                    'start_word': start,
                                    'target_word': target,
                                    'path_found': path is not None,
                                    'path_length': len(path) - 1 if path else float('inf'),
                                    'path_cost': metrics['path_cost'],
                                    'nodes_visited': metrics['nodes_visited'],
                                    'max_frontier_size': metrics['space'],
                                    'time': metrics['time']
                                })
                            except Exception as e:
                                print(f"Error in {algorithm}_{h_name} for {start}->{target}: {e}")
                    else:
                        try:
                            path, visited, metrics, _ = game.find_path(
                                start, target, algorithm=algorithm
                            )

                            results.append({
                                'simulation': i,
                                'algorithm': algorithm,
                                'word_length': word_length,
                                'start_word': start,
                                'target_word': target,
                                'path_found': path is not None,
                                'path_length': len(path) - 1 if path else float('inf'),
                                'path_cost': metrics['path_cost'],
                                'nodes_visited': metrics['nodes_visited'],
                                'max_frontier_size': metrics['space'],
                                'time': metrics['time']
                            })
                        except Exception as e:
                            print(f"Error in {algorithm} for {start}->{target}: {e}")
        except Exception as e:
            print(f"Error creating game for {word_length}-letter words: {e}")

    return pd.DataFrame(results)

def create_word_length_heatmap(df, value_col='time', title=None):
    """
    Create a heatmap to compare algorithm performance across different word lengths.

    Args:
        df: DataFrame with simulation results
        value_col: Column to use for heatmap values
        title: Chart title

    Returns:
        Altair chart
    """
    if title is None:
        title = f"{value_col.replace('_', ' ').title()} vs Word Length by Algorithm"

    # Calculate mean values for each algorithm and word length
    summary = df.groupby(['algorithm', 'word_length'])[value_col].mean().reset_index()

    # Create a heatmap
    heatmap = alt.Chart(summary).mark_rect().encode(
        x=alt.X('word_length:O', title='Word Length'),
        y=alt.Y('algorithm:N', title='Algorithm', sort=None),
        color=alt.Color(f'{value_col}:Q', title=value_col.replace('_', ' ').title(),
                      scale=alt.Scale(scheme='viridis')),
        tooltip=['algorithm', 'word_length', alt.Tooltip(f'{value_col}:Q', format='.2f')]
    ).properties(
        title=title,
        width=500,
        height=alt.Step(40)
    )

    return heatmap

def compare_heuristics_chart(df, algorithm='astar', metric='nodes_visited', title=None):
    """
    Create a chart specifically comparing different heuristics for a given algorithm.

    Args:
        df: DataFrame with simulation results
        algorithm: Algorithm to analyze (default: 'astar')
        metric: Metric to compare (default: 'nodes_visited')
        title: Chart title

    Returns:
        Altair chart
    """
    if title is None:
        title = f"Heuristic Comparison for {algorithm.upper()}: {metric.replace('_', ' ').title()}"

    # Filter for just the specified algorithm
    filtered_df = df[df['algorithm'].str.startswith(f"{algorithm}_")].copy()

    if len(filtered_df) == 0:
        print(f"No data found for algorithm {algorithm} with heuristics")
        return None

    # Extract just the heuristic name
    filtered_df['heuristic'] = filtered_df['algorithm'].str.replace(f"{algorithm}_", "", regex=False)

    # Group by heuristic and word length (if present)
    if 'word_length' in filtered_df.columns:
        summary = filtered_df.groupby(['heuristic', 'word_length'])[metric].mean().reset_index()

        # Create a grouped bar chart
        chart = alt.Chart(summary).mark_bar().encode(
            x=alt.X('heuristic:N', title='Heuristic'),
            y=alt.Y(f'{metric}:Q', title=metric.replace('_', ' ').title()),
            color='heuristic:N',
            column='word_length:N'
        ).properties(
            title=title,
            width=150,
            height=300
        )
    else:
        # For other types of problems without word_length
        summary = filtered_df.groupby(['heuristic'])[metric].mean().reset_index()

        # Create a simple bar chart
        chart = alt.Chart(summary).mark_bar().encode(
            x=alt.X('heuristic:N', title='Heuristic'),
            y=alt.Y(f'{metric}:Q', title=metric.replace('_', ' ').title()),
            color='heuristic:N'
        ).properties(
            title=title,
            width=400,
            height=300
        )

    return chart

def create_word_ladder_success_rate_chart(df, title=None):
    """
    Create a chart showing the success rate of different algorithms by word length.

    Args:
        df: DataFrame with simulation results
        title: Chart title

    Returns:
        Altair chart
    """
    if title is None:
        title = "Success Rate by Algorithm and Word Length"

    # Calculate success rate for each algorithm and word length
    success_df = df.groupby(['algorithm', 'word_length'])['path_found'].mean().reset_index()
    success_df['success_rate'] = success_df['path_found'] * 100

    # Create a bar chart for success rate
    chart = alt.Chart(success_df).mark_bar().encode(
        x=alt.X('algorithm:N', title='Algorithm'),
        y=alt.Y('success_rate:Q', title='Success Rate (%)'),
        color='algorithm:N',
        column='word_length:N',
        tooltip=['algorithm', 'word_length', alt.Tooltip('success_rate:Q', format='.1f')]
    ).properties(
        title=title,
        width=100,
        height=300
    )

    return chart

def analyze_word_ladder_performance(num_simulations=5, word_lengths=None, algorithms=None):
    """
    Run word ladder simulations and create a comprehensive analysis dashboard.

    Args:
        num_simulations: Number of simulations per word length
        word_lengths: Word lengths to test
        algorithms: Algorithms to test

    Returns:
        DataFrame with results and displays charts
    """
    from IPython.display import display

    # Run the simulations
    print("Running word ladder simulations...")
    results_df = run_word_ladder_simulations(
        num_simulations=num_simulations,
        word_lengths=word_lengths,
        algorithms=algorithms
    )

    if len(results_df) == 0:
        print("No results to analyze.")
        return results_df

    print(f"Completed {len(results_df)} algorithm runs")

    # Create and display various charts
    metrics = ['time', 'nodes_visited', 'max_frontier_size', 'path_length']

    print("\nGenerating performance charts...")
    for metric in metrics:
        chart = create_performance_charts(results_df, metric=metric)
        display(chart)

    print("\nGenerating heatmaps by word length...")
    for metric in metrics:
        heatmap = create_word_length_heatmap(results_df, value_col=metric)
        display(heatmap)

    print("\nGenerating boxplots...")
    for metric in metrics:
        boxplot = create_boxplot(results_df, value_col=metric)
        display(boxplot)

    print("\nGenerating success rate chart...")
    success_chart = create_word_ladder_success_rate_chart(results_df)
    display(success_chart)

    print("\nComparing heuristics for informed search...")
    for metric in metrics:
        heuristic_chart = compare_heuristics_chart(results_df, algorithm='astar', metric=metric)
        if heuristic_chart:
            display(heuristic_chart)

    return results_df