# Search Algorithms Playground for CS3600

This repository provides an interactive playground for exploring search algorithms as part of the CS3600 (Introduction to Artificial Intelligence) course at Georgia Tech. It contains implementations of various search strategies applied to classic AI problems.

## Overview

### Problems and Games

This repository includes implementations of three classic search problems:

1. **Graph Search**: A general implementation of search algorithms on weighted graphs, including the Romania road map example from the AIMA textbook.

2. **N-Puzzle**: The classic sliding tile puzzle (8-puzzle, 15-puzzle, etc.) where you need to rearrange numbered tiles to reach a goal configuration.

3. **Word Ladder**: A word game where you transform one word into another by changing one letter at a time, with each intermediate step being a valid word (e.g., "cat" → "cot" → "dot" → "dog").

### Implemented Algorithms

The repository includes both uninformed and informed search strategies:

- **Uninformed Search**:
  - Breadth-First Search (BFS)
  - Depth-First Search (DFS)
  - Uniform-Cost Search (UCS)

- **Informed Search**:
  - Greedy Best-First Search
  - A* Search

### Heuristics

For informed search algorithms, several heuristics are implemented:

- **For N-Puzzle**:
  - Manhattan distance
  - Misplaced tiles
  - Linear conflict

- **For Word Ladder**:
  - Hamming distance (number of differing letters)
  - Letter set difference
  - Vowel-consonant difference (intentionally poor heuristic for comparison)

- **For Graph Search**:
  - Euclidean distance
  - Manhattan distance

## How to Use This Repository

There are three ways to use this code:

### 1. Using Google Colab (Recommended for Quick Start)

A Google Colab notebook will be provided (link to be added) that allows you to run the code without any local setup.

### 2. Local Setup

To run the code locally:

1. Clone this repository:
   ```bash
   git clone <repository-url>
   ```

2. Set up the environment:

   **Using Conda (Recommended)**:
   ```bash
   conda env create -f environment.yml
   conda activate cs3600-search
   ```

   **Using pip**:
   ```bash
   pip install numpy pandas matplotlib networkx ipywidgets jupyter notebook altair
   ```

3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

4. Open `search_strategies.ipynb` to explore the interactive demonstrations.

### 3. Local Setup with Debugging

For a deeper understanding of the algorithms, you can debug the code using an IDE like VSCode:

1. Follow the local setup steps above
2. Open the repository in VSCode with the Python extension installed
3. Set breakpoints in key locations (see Debugging section below)
4. Run the Python files directly to observe algorithm execution step by step

## Files Description

- `search_strategies.ipynb`: Main notebook with examples and interactive demonstrations
- `graph_search.py`: Implementation of graph search algorithms
- `n_puzzle.py`: Implementation of the n-puzzle problem
- `word_ladder.py`: Implementation of the Word Ladder game
- `metrics.py`: Functions for measuring and visualizing algorithm performance

## Debugging the Algorithms

Debugging code is an excellent way to understand the algorithms and their implementation details. This guide will help you set up breakpoints directly in the main files to observe how the algorithms work step by step.

### Setting Up a Debug Environment

#### Using VSCode (Recommended)

1. Open the project folder in VSCode
2. Install the Python extension if you haven't already
3. Go to the Debug tab (or press `Ctrl+Shift+D` / `Cmd+Shift+D`)
4. Create a `launch.json` file if it doesn't exist yet, with a Python configuration

![VSCode Debug Setup](screenshots/vscode_debug_setup.png)

### Debugging graph_search.py

1. Open `graph_search.py` in your editor
2. Scroll to the end of the file where the `if __name__ == "__main__":` section is
3. This section already has example code that runs the algorithm on the Romania map
4. Set breakpoints at key points in the `graph_search` function:
   - Line ~210: At the beginning of the function to observe initialization
   - Line ~270: Inside the main `while frontier:` loop to see each iteration
   - Line ~290: Where neighbors are processed and added to the frontier
   - Line ~240: Where the goal is found and the path is returned

![Setting Breakpoints](screenshots/setting_breakpoints.png)

5. Run the file in debug mode:
   - In VSCode: Press F5 or click the green play button in the Debug tab
   - This will execute the code in `if __name__ == "__main__":` section and hit your breakpoints

### Debugging n_puzzle.py

1. Open `n_puzzle.py` in your editor
2. Go to the `if __name__ == "__main__":` section at the bottom
3. This section already contains code that creates and solves an 8-puzzle
4. Set breakpoints in the `n_puzzle_search` function:
   - At the beginning of the function to observe initialization
   - In the main search loop to observe state expansion
   - Where goal states are checked
   - Where new states are added to the frontier

5. Run the file in debug mode to hit your breakpoints and observe the algorithm

### Debugging word_ladder.py

1. Open `word_ladder.py` in your editor
2. Go to the `if __name__ == "__main__":` section at the bottom
3. The section creates a WordLadderGame and finds a path from "cat" to "dog"
4. Set breakpoints in these key locations:
   - In the `_create_word_graph` method to understand graph construction
   - In the `find_path` method which calls the general graph search function
   - In the `_differs_by_one_letter` method to see how neighbors are determined

5. Run the file in debug mode to hit your breakpoints

### Key Variables to Watch

When debugging these algorithms, pay attention to these important variables:

#### In graph_search.py:
- `frontier`: This shows the nodes to be explored next (queue, stack, or priority queue)
- `visited`: The set of nodes already explored
- `current`: The node being processed in the current iteration
- `path`: The current path being built
- `cost`: The cost of the path so far

#### In n_puzzle.py:
- `frontier`: The states to be explored next
- `visited_states`: States that have been visited already
- `current_state`: The puzzle state being examined
- `actions`: Possible moves from the current state
- `successors`: New states generated from the current state

#### In word_ladder.py:
- `graph`: The word graph connecting words that differ by one letter
- `words`: The set of valid words in the dictionary
- `path`: The sequence of words from start to target

![Watching Variables](screenshots/watching_variables.png)

### Understanding Algorithm Differences

By setting breakpoints and watching the execution, you can understand the key differences between algorithms:

#### BFS vs DFS
- In BFS (`algorithm='bfs'`): Watch how frontier acts as a queue (FIFO)
- In DFS (`algorithm='dfs'`): Watch how frontier acts as a stack (LIFO)
- Notice how BFS explores breadth-first (all neighbors before moving deeper)
- Notice how DFS explores depth-first (follows one path deeply before backtracking)

#### UCS, Greedy, and A*
- In UCS (`algorithm='ucs'`): Observe how nodes are prioritized by path cost
- In Greedy (`algorithm='greedy'`): See how the heuristic alone determines priority
- In A* (`algorithm='astar'`): Watch the combination of path cost and heuristic at work

![Algorithm Comparison](screenshots/algorithm_comparison.png)

### Debugging Tips

- Use the "Step Into" feature to go into function calls
- Use "Step Over" to execute a line without diving into function details
- Use "Continue" to run until the next breakpoint
- Add "Watch" expressions to monitor complex expressions or data structures
- Use conditional breakpoints for specific scenarios (right-click on a breakpoint)
- Use the Debug Console to evaluate expressions during a debugging session

The `screenshots` folder contains reference images for these debugging steps. You can replace them with your own screenshots as you debug the code.

## License

MIT License

Copyright (c) 2023 Georgia Institute of Technology

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.