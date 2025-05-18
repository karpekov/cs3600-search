# Search Algorithms

This directory contains implementations of various graph search algorithms and the n-puzzle problem.

## Setting Up the Environment

To set up the environment for running the code in this directory, follow these steps:

### Using Conda (Recommended)

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution) if you don't have it already.

2. Create a new environment from the provided `environment.yml` file:
   ```bash
   conda env create -f environment.yml
   ```

3. Activate the environment:
   ```bash
   conda activate cs3600-search
   ```

4. Launch Jupyter Notebook to open the search_strategies.ipynb:
   ```bash
   jupyter notebook
   ```

### Using pip

If you prefer to use pip, you can install the required packages with:

```bash
pip install numpy pandas matplotlib networkx ipywidgets jupyter notebook altair
```

## Files Description

- `search_strategies.ipynb`: Main notebook with examples and interactive demonstrations
- `graph_search.py`: Implementation of graph search algorithms
- `n_puzzle.py`: Implementation of the n-puzzle problem
- `metrics.py`: Functions for measuring and visualizing algorithm performance