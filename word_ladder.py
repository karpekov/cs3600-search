import time
import heapq
from collections import deque
import networkx as nx
import matplotlib.pyplot as plt
import os
import string
import numpy as np
from IPython.display import display, clear_output
import ipywidgets as widgets

# Import the graph search algorithm from graph_search.py
from graph_search import graph_search, visualize_graph

# Try to import nltk; if not available, we'll handle it gracefully
try:
    import nltk
    from nltk.corpus import words as nltk_words
except ImportError:
    print("NLTK not found. Will use a smaller word list or download NLTK.")

class WordLadderGame:
    """
    Implementation of the Word Ladder game where the goal is to transform one word into another
    by changing one letter at a time, and only using valid words.
    """

    def __init__(self, word_length=4, use_nltk=True):
        """
        Initialize the Word Ladder game.

        Args:
            word_length: Length of words to use (3, 4, or 5)
            use_nltk: Whether to use NLTK for word corpus or a smaller built-in list
        """
        self.word_length = word_length

        # Check if word_length is valid
        if word_length not in [3, 4, 5]:
            raise ValueError("Word length must be 3, 4, or 5")

        # Get word list
        self.words = self._get_words(word_length, use_nltk)

        # Create graph
        self.graph = self._create_word_graph()

    @classmethod
    def generate_word_list_files(cls):
        """
        Generate word list files for 3, 4, and 5 letter words.
        Files are only generated if they don't already exist.
        """
        # Create the word_lists directory if it doesn't exist
        os.makedirs('word_lists', exist_ok=True)

        # Check which files need to be generated
        for length in [3, 4, 5]:
            file_path = os.path.join('word_lists', f'{length}_letter_words.txt')

            # Skip if file already exists
            if os.path.exists(file_path):
                print(f"Word list file for {length}-letter words already exists, skipping.")
                continue

            print(f"Generating word list file for {length}-letter words...")

            # Create a temporary game instance to get words
            game = cls(word_length=length, use_nltk=True)

            # Write the words to file
            with open(file_path, 'w') as f:
                for word in sorted(game.words):
                    f.write(f"{word}\n")

            print(f"Created {file_path} with {len(game.words)} words.")

    def _get_words(self, length, use_nltk):
        """
        Get a list of valid English words of the specified length.

        Args:
            length: Length of words to retrieve
            use_nltk: Whether to use NLTK corpus or a smaller built-in list

        Returns:
            Set of valid words
        """
        # First try to load from the word list file
        file_path = os.path.join('word_lists', f'{length}_letter_words.txt')
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    words = set(word.strip().lower() for word in f)
                    if len(words) > 10:  # Arbitrary threshold for a valid word list
                        print(f"Loaded {len(words)} {length}-letter words from file")
                        return words
                    print(f"Word list file too small, trying other methods")
            except Exception as e:
                print(f"Error reading word list file: {e}")

        # If no file or file too small, try NLTK if requested
        if use_nltk:
            try:
                # Try to use NLTK
                nltk.download('words', quiet=True)
                word_list = set(w.lower() for w in nltk_words.words() if len(w) == length and w.isalpha())

                # If we got a reasonable number of words, return them
                if len(word_list) > 100:  # Arbitrary threshold
                    print(f"Using {len(word_list)} words from NLTK")
                    return word_list

                print(f"NLTK word list too small ({len(word_list)} words). Using built-in list.")
            except Exception as e:
                print(f"Error using NLTK: {e}. Using built-in list.")

        # If all else fails, use built-in lists
        print(f"Using built-in list for {length}-letter words")
        if length == 3:
            return {'cat', 'bat', 'hat', 'mat', 'sat', 'rat', 'pat', 'fat', 'vat',
                   'cot', 'cut', 'car', 'cap', 'can', 'cab', 'cad', 'caw', 'cay',
                   'dog', 'fog', 'log', 'bog', 'cog', 'hog', 'jog', 'fix', 'six',
                   'mix', 'pit', 'hit', 'fit', 'sit', 'lit', 'bit', 'win', 'tin',
                   'bin', 'fin', 'pin', 'sin', 'run', 'sun', 'fun', 'gun', 'bun'}
        elif length == 4:
            return {'word', 'play', 'game', 'book', 'work', 'time', 'year',
                   'life', 'home', 'room', 'door', 'open', 'case', 'make',
                   'take', 'give', 'find', 'look', 'tell', 'come', 'know',
                   'think', 'good', 'time', 'turn', 'help', 'talk', 'walk'}
        elif length == 5:
            return {'about', 'above', 'after', 'again', 'along', 'begin',
                   'think', 'water', 'first', 'house', 'light', 'night',
                   'place', 'sound', 'world', 'write', 'young'}

        return set()  # Empty set as fallback

    def _create_word_graph(self):
        """
        Create a graph where nodes are words and edges connect words
        that differ by exactly one letter.

        Returns:
            NetworkX graph of connected words
        """
        G = nx.Graph()
        G.add_nodes_from(self.words)

        # Add edges between words that differ by one letter
        word_list = list(self.words)
        for i in range(len(word_list)):
            word1 = word_list[i]
            for j in range(i+1, len(word_list)):
                word2 = word_list[j]
                if self._differs_by_one_letter(word1, word2):
                    G.add_edge(word1, word2, weight=1)

        return G

    def _differs_by_one_letter(self, word1, word2):
        """
        Check if two words differ by exactly one letter.

        Args:
            word1: First word
            word2: Second word

        Returns:
            True if the words differ by exactly one letter, False otherwise
        """
        if len(word1) != len(word2):
            return False

        differences = sum(1 for a, b in zip(word1, word2) if a != b)
        return differences == 1

    def get_neighbors(self, word):
        """
        Get all valid words that differ by one letter from the given word.

        Args:
            word: The word to find neighbors for

        Returns:
            List of neighboring words
        """
        return list(self.graph.neighbors(word))

    def is_valid_word(self, word):
        """
        Check if a word is in the valid word list.

        Args:
            word: Word to check

        Returns:
            True if the word is valid, False otherwise
        """
        return word in self.words

    def is_valid_path(self, path):
        """
        Check if a path is valid (each step changes only one letter).

        Args:
            path: List of words

        Returns:
            True if the path is valid, False otherwise
        """
        if not path or len(path) < 2:
            return True

        for i in range(len(path) - 1):
            if not self._differs_by_one_letter(path[i], path[i+1]):
                return False

        return True

    def find_path(self, start_word, target_word, algorithm='bfs', heuristic=None):
        """
        Find a path from start_word to target_word using the specified algorithm.

        Args:
            start_word: Starting word
            target_word: Target word
            algorithm: Search algorithm to use
            heuristic: Heuristic function for informed search

        Returns:
            Path (list of words), visited words, metrics, and search states
        """
        if start_word not in self.words:
            raise ValueError(f"Start word '{start_word}' is not a valid {self.word_length}-letter word")
        if target_word not in self.words:
            raise ValueError(f"Target word '{target_word}' is not a valid {self.word_length}-letter word")

        # Use the graph_search function from graph_search.py
        path, visited, metrics, states = graph_search(
            self.graph, start_word, target_word,
            algorithm=algorithm,
            heuristic=heuristic
        )

        return path, visited, metrics, states

    def visualize_word_graph(self, highlighted_path=None, highlighted_nodes=None, frontier=None):
        """
        Visualize the word graph with optional highlighted path, nodes, and frontier.

        Args:
            highlighted_path: List of words to highlight (path)
            highlighted_nodes: List of words to highlight (visited)
            frontier: List of words in the frontier

        Returns:
            Matplotlib figure
        """
        # The word graph can be huge, so we'll create a subgraph with just the relevant words
        if highlighted_path or highlighted_nodes or frontier:
            relevant_words = set()

            if highlighted_path:
                relevant_words.update(highlighted_path)
            if highlighted_nodes:
                relevant_words.update(highlighted_nodes)
            if frontier:
                relevant_words.update(frontier)

            # Add immediate neighbors of all relevant words
            neighbors = set()
            for word in relevant_words:
                neighbors.update(self.graph.neighbors(word))
            relevant_words.update(neighbors)

            # Create subgraph
            subgraph = self.graph.subgraph(list(relevant_words))
        else:
            # If no specifics, just show a sample of the graph
            sample_size = min(100, len(self.words))
            sample_words = list(self.words)[:sample_size]
            subgraph = self.graph.subgraph(sample_words)

        # Use a layout algorithm appropriate for larger graphs
        pos = nx.spring_layout(subgraph, seed=42)

        plt.figure(figsize=(12, 10))

        # Draw the basic graph
        nx.draw_networkx_nodes(subgraph, pos, node_size=500, node_color='lightblue')
        nx.draw_networkx_labels(subgraph, pos, font_size=10)
        nx.draw_networkx_edges(subgraph, pos, width=1.0, alpha=0.3)

        # Highlight visited nodes if provided
        if highlighted_nodes:
            nodes_to_highlight = [node for node in highlighted_nodes if node in subgraph]
            if nodes_to_highlight:
                nx.draw_networkx_nodes(subgraph, pos, nodelist=nodes_to_highlight, node_color='yellow', node_size=500)

        # Highlight frontier nodes if provided
        if frontier:
            frontier_nodes = [node for node in frontier if node in subgraph]
            if frontier_nodes:
                nx.draw_networkx_nodes(subgraph, pos, nodelist=frontier_nodes, node_color='orange', node_size=500)

        # Highlight path if provided
        if highlighted_path:
            path_edges = [(highlighted_path[i], highlighted_path[i+1]) for i in range(len(highlighted_path)-1)
                          if highlighted_path[i] in subgraph and highlighted_path[i+1] in subgraph]
            if path_edges:
                nx.draw_networkx_edges(subgraph, pos, edgelist=path_edges, width=3, edge_color='red')

        plt.title(f"Word Ladder Graph - {self.word_length}-letter words", fontsize=16)
        plt.axis('off')
        plt.tight_layout()

        return plt.gcf()

    def visualize_word_change(self, start_word, current_path, step=0):
        """
        Visualize word transformation as a series of letter blocks with changes highlighted.

        Args:
            start_word: The starting word
            current_path: The current path of words
            step: The current step in the search

        Returns:
            Matplotlib figure
        """
        word_length = len(start_word)
        num_words = len(current_path)

        fig, ax = plt.subplots(figsize=(2 * word_length, 1.5 * num_words))

        # Draw a grid of squares, one row per word in the path
        for i, word in enumerate(current_path):
            for j, letter in enumerate(word):
                # Determine if this letter changed from the previous word
                changed = False
                if i > 0 and j < len(current_path[i-1]) and letter != current_path[i-1][j]:
                    changed = True

                # Draw the square
                rect = plt.Rectangle((j, -i), 0.9, 0.9,
                                    color='lightgreen' if changed else 'lightblue')
                ax.add_patch(rect)

                # Add the letter
                ax.text(j + 0.45, -i + 0.45, letter.upper(),
                        horizontalalignment='center', verticalalignment='center',
                        fontsize=14, fontweight='bold')

        ax.set_xlim(-0.1, word_length)
        ax.set_ylim(-num_words, 1)
        ax.set_aspect('equal')
        ax.axis('off')

        plt.title(f"Word Ladder: Step {step}", fontsize=16)
        plt.tight_layout()

        return fig

# Define heuristic functions for word ladder

def hamming_distance(word1, word2):
    """
    Count the number of positions where the letters differ.
    This is a good heuristic for word ladder.

    Args:
        word1: First word
        word2: Second word

    Returns:
        Number of positions where letters differ
    """
    return sum(1 for a, b in zip(word1, word2) if a != b)

def letter_set_difference(word1, word2):
    """
    Count how many letters in word1 are not in word2.
    This is a weak heuristic for word ladder.

    Args:
        word1: First word
        word2: Second word

    Returns:
        Number of letters in word1 that are not in word2
    """
    return sum(1 for letter in word1 if letter not in word2)

def vowel_consonant_difference(word1, word2):
    """
    A deliberately poor heuristic that counts difference in vowel/consonant patterns.

    Args:
        word1: First word
        word2: Second word

    Returns:
        A measure based on vowel/consonant patterns
    """
    vowels = set('aeiou')

    def is_vowel(letter):
        return letter.lower() in vowels

    word1_pattern = [is_vowel(letter) for letter in word1]
    word2_pattern = [is_vowel(letter) for letter in word2]

    return sum(1 for p1, p2 in zip(word1_pattern, word2_pattern) if p1 != p2)

def create_word_ladder_demo(game, start_word=None, target_word=None):
    """
    Create an interactive demo of the word ladder game search algorithms.

    Args:
        game: WordLadderGame instance
        start_word: Optional starting word (if None, will be chosen from dropdown)
        target_word: Optional target word (if None, will be chosen from dropdown)
    """
    # Sample words to put in the dropdown as examples
    word_length = game.word_length
    sample_words = list(game.words)[:100]  # First 100 words

    # Ensure start_word and target_word are in the list if provided
    if start_word and start_word in game.words and start_word not in sample_words:
        sample_words[0] = start_word

    if target_word and target_word in game.words and target_word not in sample_words:
        if len(sample_words) > 1:
            sample_words[1] = target_word
        else:
            sample_words.append(target_word)

    if not start_word:
        start_word = sample_words[0] if sample_words else ""
    if not target_word:
        target_word = sample_words[-1] if len(sample_words) > 1 else ""

    # Create interactive widgets
    algorithm_dropdown = widgets.Dropdown(
        options=['BFS', 'DFS', 'UCS', 'Greedy', 'Astar'],
        value='BFS',
        description='Algorithm:',
        style={'description_width': 'initial'}
    )

    start_word_dropdown = widgets.Dropdown(
        options=sample_words,
        value=start_word,
        description='Start Word:',
        style={'description_width': 'initial'}
    )

    target_word_dropdown = widgets.Dropdown(
        options=sample_words,
        value=target_word,
        description='Target Word:',
        style={'description_width': 'initial'}
    )

    heuristic_dropdown = widgets.Dropdown(
        options=['Hamming Distance', 'Letter Set Difference', 'Vowel-Consonant (Poor)'],
        value='Hamming Distance',
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

    view_toggle = widgets.ToggleButtons(
        options=['Word Changes', 'Graph View'],
        description='View:',
        disabled=False,
        button_style='',
        style={'description_width': 'initial'}
    )

    output = widgets.Output()
    graph_output = widgets.Output()

    # Store search results
    search_results = {}

    def update_visualization(step):
        with graph_output:
            clear_output(wait=True)
            if not search_results:
                if view_toggle.value == 'Graph View':
                    game.visualize_word_graph()
                    plt.show()
                else:
                    game.visualize_word_change(start_word_dropdown.value, [start_word_dropdown.value])
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

            if view_toggle.value == 'Graph View':
                title = f"Step {step+1}/{len(states['steps'])}: {algorithm_dropdown.value} Search"
                fig = game.visualize_word_graph(
                    highlighted_path=path,
                    highlighted_nodes=visited,
                    frontier=frontier
                )
                plt.show()
            else:
                fig = game.visualize_word_change(
                    start_word_dropdown.value,
                    path,
                    step=step+1
                )
                plt.show()

            # Display metadata below the visualization
            print(f"Current word: {current}")
            print(f"Path so far: {' -> '.join(path)}")
            print(f"Path cost: {len(path) - 1}")
            print(f"Visited words: {len(visited)}")
            print(f"Frontier size: {len(frontier)}")

            # If we found the goal
            if current == target_word_dropdown.value:
                print("\n🎯 Goal reached!")
                metrics = search_results.get('metrics')
                if metrics:
                    print(f"\nSearch Metrics:")
                    print(f"Time taken: {metrics['time']:.6f} seconds")
                    print(f"Max frontier size: {metrics['space']}")
                    print(f"Total nodes visited: {metrics['nodes_visited']}")

    def on_step_change(change):
        if change['name'] == 'value':
            update_visualization(change['new'])

    def on_view_toggle(change):
        if change['name'] == 'value':
            update_visualization(step_slider.value)

    def on_run_button_clicked(b):
        with output:
            clear_output()
            algorithm = algorithm_dropdown.value.lower()
            start = start_word_dropdown.value
            target = target_word_dropdown.value
            heuristic_name = heuristic_dropdown.value

            # Select appropriate heuristic
            heuristic = None
            if algorithm in ['greedy', 'astar']:
                if heuristic_name == 'Hamming Distance':
                    heuristic = hamming_distance
                elif heuristic_name == 'Letter Set Difference':
                    heuristic = letter_set_difference
                elif heuristic_name == 'Vowel-Consonant (Poor)':
                    heuristic = vowel_consonant_difference

            print(f"Running {algorithm} search from '{start}' to '{target}'...")

            try:
                # Run the search
                path, visited, metrics, states = game.find_path(
                    start, target,
                    algorithm=algorithm,
                    heuristic=heuristic
                )

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
                    print(f"✓ Path found! Length: {len(path) - 1}, Cost: {metrics['path_cost']}")
                    print(f"Path: {' -> '.join(path)}")
                else:
                    print("✗ No path found!")

                print(f"Words visited: {metrics['nodes_visited']}")
                print(f"Max frontier size: {metrics['space']}")
                print(f"Time taken: {metrics['time']:.6f} seconds")
                print("\nUse the slider or Next Step button to step through the search process.")

                # Update the visualization
                update_visualization(0)

            except Exception as e:
                print(f"Error during search: {e}")

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
    view_toggle.observe(on_view_toggle, names='value')

    # Initial visualization display
    with graph_output:
        if view_toggle.value == 'Graph View':
            game.visualize_word_graph()
            plt.show()
        else:
            game.visualize_word_change(start_word_dropdown.value, [start_word_dropdown.value])
            plt.show()

    # Layout the widgets
    top_row = widgets.HBox([algorithm_dropdown, start_word_dropdown, target_word_dropdown])
    middle_row = widgets.HBox([heuristic_dropdown, view_toggle])
    control_row = widgets.HBox([run_button, step_slider, next_step_button])

    # Display everything
    display(widgets.VBox([
        top_row,
        middle_row,
        control_row,
        output,
        graph_output
    ]))



# If running directly, test with a sample word ladder
if __name__ == "__main__":
    # Generate word list files if needed
    WordLadderGame.generate_word_list_files()

    # Create a word ladder game with 3-letter words
    game = WordLadderGame(word_length=3)

    # Find a path from "cat" to "dog"
    path, visited, metrics, states = game.find_path("cat", "dog", algorithm="astar", heuristic=hamming_distance)

    if path:
        print(f"Path found: {' -> '.join(path)}")
        print(f"Path length: {len(path) - 1}")
        print(f"Nodes visited: {metrics['nodes_visited']}")
    else:
        print("No path found")