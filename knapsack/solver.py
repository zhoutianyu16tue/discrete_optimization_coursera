#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
import numpy as np
from time import time
Item = namedtuple("Item", ['index', 'value', 'weight'])
import sys

def solve_lb_1(items, capacity):
    items = sorted(items, key=lambda i:i.weight)

    curr_val, curr_weight = 0, 0
    taken = np.zeros(len(items), dtype=np.int)
    for item in items:
        if curr_weight + item.weight <= capacity:
            curr_weight += item.weight
            curr_val += item.value
            taken[item.index] = 1
        else:
            break

    return curr_val, taken

def solve_lb_2(items, capacity):
    items = sorted(items, key=lambda i:i.value, reverse=True)

    curr_val, curr_weight = 0, 0
    taken = np.zeros(len(items), dtype=np.int)
    for item in items:
        if curr_weight + item.weight <= capacity:
            curr_weight += item.weight
            curr_val += item.value
            taken[item.index] = 1
        else:
            break

    return curr_val, taken

def solve_calc_linear_relax_value(items, capacity):

    curr_weight = 0
    curr_val = 0
    greedy_lb = 0
    for item in items:
        if curr_weight + item.weight <= capacity:
            curr_weight += item.weight
            curr_val += item.value
        else:
            greedy_lb = curr_val
            curr_val += item.value * (capacity - curr_weight) / item.weight
            break

    return int(greedy_lb), int(curr_val)



def solve_recur_helper(curr_highest_val, curr_val, curr_weight, item_selected, items, estimate, capacity, greedy_lb, ret):

    if curr_weight > capacity:
        return 

    if len(items) == 0:

        if curr_val >= greedy_lb and curr_val > curr_highest_val[-1]:
            curr_highest_val[-1] = curr_val
            # print("One solution has been found")
            # print(curr_val, curr_weight, item_selected, estimate, curr_highest_val)
            # print()
            ret.clear()
            ret[tuple(item_selected)] = curr_highest_val[-1]
        return

    # Select the current item

    solve_recur_helper(curr_highest_val, curr_val+items[0].value, curr_weight+items[0].weight, item_selected + [1], items[1:], estimate, capacity, greedy_lb, ret)

    # Don't select the current item, need to calculate new estimate
    _, new_estimate = solve_calc_linear_relax_value(items[1:], capacity-curr_weight)

    if (new_estimate + curr_val) < greedy_lb or (new_estimate + curr_val) < curr_highest_val[-1]:
        # print("One branch has been pruned")
        # print("new estimate", new_estimate, greedy_lb)
        # print(curr_val, curr_weight, item_selected, estimate, curr_highest_val)
        # print()
        return

    solve_recur_helper(curr_highest_val, curr_val, curr_weight, item_selected + [0], items[1:], new_estimate+curr_val, capacity, greedy_lb, ret)

def format_return_str(value, taken, opt):
    output_data = str(int(value)) + ' ' + str(opt) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        if int(parts[1]) > capacity or int(parts[0])<=0:
            continue
        items.append(Item(i-1, int(parts[0]), int(parts[1])))
    # print(items)

    # # DP solution
    # print(len(items), capacity)
    # greedy_lb_0, taken_0, estimate = solve_calc_linear_relax_value(items, capacity)
    # greedy_lb_1, taken_1 = solve_lb_1(items, capacity)
    # greedy_lb_2, taken_2 = solve_lb_2(items, capacity)
    # print("greedy_lb_0", greedy_lb_0)
    # print("greedy_lb_1", greedy_lb_1)
    # print("greedy_lb_2", greedy_lb_2)
    # print("upper_bound", estimate)
    # solution = max((greedy_lb_0, taken_0), (greedy_lb_1, taken_1), (greedy_lb_2, taken_2), key=lambda t:t[0])
    # # print(item_count)
    # # print(solution[0], solution[1])
    # assert len(solution[1]) == item_count
    # return format_return_str(solution[0], solution[1], 0)
    # print("Running DP solution")
    # dp_start = time()
    # memory = np.zeros(capacity+1)
    # item_count = len(items)
    # memory[items[0].weight:] = items[0].value
    # for i in range(1, item_count): 
    #     for w in range(capacity, items[i].weight-1, -1):

    #         memory[w] = max(memory[w-items[i].weight] + items[i].value, memory[w])

    # dp_end = time()
    # print("DP takes %f seconds" % (dp_end - dp_start))
    # print(format_return_str(memory[-1], [0]*len(items), 1))
    # print()
    # return format_return_str(memory[-1], [0]*len(items), 1)

    # Branch and bounding solution
    print("Running branch and bounding solution")
    # print("Item length", len(items))
    sys.setrecursionlimit(12000)
    bb_start = time()
    
    greedy_lb_0, estimate = solve_calc_linear_relax_value(items, capacity)
    greedy_lb_1,_ = solve_lb_1(items, capacity)
    greedy_lb_2,_ = solve_lb_2(items, capacity)
    # print("greedy_lb_0", greedy_lb_0)
    # print("greedy_lb_1", greedy_lb_1)
    # print("greedy_lb_2", greedy_lb_2)
    # print("upper_bound", estimate)
    greedy_lb = max(greedy_lb_0, greedy_lb_1, greedy_lb_2)
    # print(item_count)
    # print(greedy_lb, estimate)
    items = sorted(items, key=lambda i:i.value/i.weight, reverse=True)
    ret = {}
    solve_recur_helper([0], 0, 0, [], items, estimate, capacity, greedy_lb, ret)
    # print(ret)
    bb_end = time()
    print("Branch and bounding takes %f seconds" % (bb_end - bb_start))
    # print()
    # assert len(ret) == 1

    taken = [t[1] for t in sorted(zip(items, list(ret.keys())[0]), key=lambda t:t[0].index)]
    value = list(ret.values())[0]

    # print(format_return_str(value, taken))
    return format_return_str(value, taken, 1)




if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

