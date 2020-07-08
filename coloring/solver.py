#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from itertools import groupby

def format_output(solution, obj):
    output_data = str(obj) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))
    return output_data

def greedy_bfs(node_count, adj_list, degrees):
    visited = np.zeros(node_count, dtype=np.int)
    coloring = np.zeros(node_count, dtype=np.int)
    coloring[:] = -1
    no_color_used = 1
    first_node = np.argmax(degrees)
    coloring[first_node] = 0
    queue = [first_node]
    visited[first_node] = 1

    while len(queue) != 0:

        assert len(queue) == len(set(queue))
        curr_node = queue[-1]
        queue = queue[:-1]

        for neighbor in adj_list[curr_node]:

            if not visited[neighbor]:
                assert coloring[neighbor] == -1
                visited[neighbor] = 1
                queue.append(neighbor)

                neighbor_coloring = coloring[list(adj_list[neighbor])]
                used_colors = set(neighbor_coloring[neighbor_coloring>=0])
                assert len(used_colors) <= no_color_used
                
                if len(used_colors) == no_color_used:
                    coloring[neighbor] = no_color_used
                    no_color_used += 1
                else:
                    assert used_colors.issubset(set(np.arange(no_color_used)))
                    candidate_colors = set(np.arange(no_color_used)) - used_colors
                    coloring[neighbor] = min(candidate_colors)

    return coloring, no_color_used

def greedy_deg_first(node_count, adj_list, degrees):
    coloring = np.zeros(node_count, dtype=np.int)
    coloring[:] = -1
    no_color_used = 0
    for node, deg in sorted(zip(np.arange(node_count), degrees), key=lambda t:t[1], reverse=True):
        if coloring[node] != -1:
            continue

        neighbor_coloring = coloring[list(adj_list[node])]
        used_colors = set(neighbor_coloring[neighbor_coloring>=0])
        assert len(used_colors) <= no_color_used
        
        if len(used_colors) == no_color_used:
            coloring[node] = no_color_used
            no_color_used += 1
        else:
            assert used_colors.issubset(set(np.arange(no_color_used)))
            candidate_colors = set(np.arange(no_color_used)) - used_colors
            coloring[node] = min(candidate_colors)

    return coloring, no_color_used

def greedy_dynamic_neighbor_color_first(node_count, adj_list, degrees):
    
    coloring = np.zeros(node_count, dtype=np.int)
    coloring[:] = -1
    neighbor_coloring_count = np.zeros(node_count, dtype=np.int)
    no_color_used = 1
    coloring[np.argmax(degrees)] = 0

    neighbor_coloring_count[list(adj_list[np.argmax(degrees)])] = 1

    while len(coloring[coloring < 0]) > 0:
        a = list(zip(np.arange(node_count), degrees, neighbor_coloring_count, coloring))
        a = [i for i in a if i[3] == -1]
        a = sorted(a, key=lambda t:t[2], reverse=True)
        a = sorted([i for i in a if a[0][2] == i[2]], key=lambda t:t[1], reverse=True)
        node = a[0][0]

        neighbor_coloring = coloring[list(adj_list[node])]

        used_colors = set(neighbor_coloring[neighbor_coloring>=0])
        assert len(used_colors) <= no_color_used
        
        if len(used_colors) == no_color_used:
            coloring[node] = no_color_used
            no_color_used += 1
        else:
            assert used_colors.issubset(set(np.arange(no_color_used)))
            candidate_colors = set(np.arange(no_color_used)) - used_colors
            coloring[node] = min(candidate_colors)

        for neighbor in adj_list[node]:
            a = coloring[list(adj_list[neighbor])]
            neighbor_coloring_count[neighbor] = len(set(a[a >= 0]))
    
    return coloring, no_color_used

def greedy_pruning_node_color(node_count, adj_list, degrees):

    starting_arr = np.append(np.array(sorted(list(zip(np.arange(node_count), degrees)), key=lambda t:t[1])), np.arange(node_count-1, -1, -1).reshape(-1, 1), axis=1)
    coloring = np.array([t[2] for t in sorted(starting_arr, key=lambda t:t[0])])
    candidate_colors = set(np.arange(node_count))

    neighbor_coloring_count = np.zeros(node_count, dtype=np.int)

    for node in np.arange(node_count):
        neighbor_coloring_count[node] = len(set(coloring[list(adj_list[node])]))
    print(neighbor_coloring_count)

    print(neighbor_coloring_count[[t[0] for t in starting_arr]])

    print(starting_arr)
    for t in starting_arr:
        node, curr_color = t[0], t[2]
        neighbor_coloring = coloring[list(adj_list[node])]

        coloring[node] = min(candidate_colors - set(neighbor_coloring))

    print(coloring)
    return coloring, len(set(coloring))


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    edges = []
    degrees = [0] * node_count
    adj_list = {}
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        n1, n2 = int(parts[0]), int(parts[1])
        edges.append((n1, n2))
        degrees[n1] += 1
        degrees[n2] += 1

        if n1 in adj_list:
            adj_list[n1].add(n2)
        else:
            adj_list[n1] = set([n2])

        if n2 in adj_list:
            adj_list[n2].add(n1)
        else:
            adj_list[n2] = set([n1])

    ret = [greedy_bfs(node_count, adj_list, degrees), \
            greedy_deg_first(node_count, adj_list, degrees), \
            greedy_dynamic_neighbor_color_first(node_count, adj_list, degrees)]

    print(ret)
    ret = min(ret, key=lambda t:t[1])

    print("=" * 50)
    solution = np.array(ret[0])

    for node in range(node_count):
        print(node, solution[node], degrees[node], set(solution[list(adj_list[node])]))

    print("=" * 50)
    return format_output(ret[0], ret[1])


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')

