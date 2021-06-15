# Description

This is a python program that solves a 2x2x2 Rubik's cube using astar, bfs, dfs, ids, dlastar, and random walk. It was developed as a part of my Artificial Intelligence class and then later improved upon. 

# Usage

### Command Syntax

run.sh is used to pass commands and arugments to the python program. 

```python
sh run.sh <command> <scramble>
```
### Supported commands

The following commands are supported: 

* bfs - breadth-first search
* dfs - depth-first search
* ids - iterative depth limited search
* random - random walk down the tree
* astar - astar using a pattern database heuristic
* astarMD - astar using Manhattan Distance
* idastar - iterative depth limited astar

### Scramble

The scramble is passed in the format of following moves:

* R/R' - rotates the right face clockwise/counter-clockwise
* L/L' - rotates the left face clockwise/counter-clockwise
* U/U' - rotates the upper face clockwise/counter-clockwise
* D/D' - rotates the bottom face clockwise/counter-clockwise
* F/F' - rotates the front face clockwise/counter-clockwise
* B/B' - rotates the back face clockwise/counter-clockwise

Example - "F R U L' D B"

### Credits

I had additional help from [Kevin Karnani](https://github.com/kevinkarnani) and [Anthony Goncharenko](https://github.com/AnthonyGoncharenko) for improving and optimizing the code after the assignment. 

