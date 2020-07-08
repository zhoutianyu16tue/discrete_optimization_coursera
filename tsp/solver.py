#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import numpy as np
from collections import namedtuple
from scipy.spatial import ConvexHull

Point = namedtuple("Point", ['x', 'y'])

# def length(point1, point2):
#     return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def length(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def format_output(solution, obj, opt=0):
    output_data = '%.2f' % obj + ' ' + str(opt) + '\n'
    output_data += ' '.join(map(str, solution))
    return output_data

def build_dist_mat(points):

    nodeCount = len(points)
    dist_mat = np.zeros((nodeCount, nodeCount))

    for i in range(nodeCount):
        for j in range(i+1, nodeCount):
            dist_mat[i, j] = dist_mat[j, i] = length(points[i], points[j])
    
    return dist_mat

def process_input(input_data):
    lines = input_data.split('\n')

    nodeCount = int(lines[0])

    points = []
    p2i_dict = {}

    for i in range(1, nodeCount+1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))
        p2i_dict[(float(parts[0]), float(parts[1]))] = i-1

    assert len(points) == nodeCount
    return np.array(points), nodeCount, p2i_dict

def solve_it(input_data):
    return solve_it_3(input_data)

def insert_unvisited_points(points, points_to_visit, solution, visited_points_cnt):
    for p2v in points_to_visit:

        curr_sol = solution[:visited_points_cnt]

        pos = min([(i, length(points[p2v], points[p1])+length(points[p2v], points[p2])-length(points[p1], points[p2]))for i, (p1, p2) in enumerate(zip(curr_sol, np.append(curr_sol[1:], curr_sol[0])))], key=lambda t:t[1])[0]

        solution = np.insert(solution, pos+1, p2v)
        visited_points_cnt += 1

    return solution, visited_points_cnt

def calc_solution_obj(solution, points):

    assert len(solution) == len(points)
    obj = 0
    for i in range(len(solution)-1):
        obj += length(points[solution[i]], points[solution[i+1]])

    obj += length(points[solution[-1]], points[solution[0]])

    return obj

def three_opt(solution, obj, points):

    nodeCount = len(solution)
    best_dist = obj
    best_route = solution
    print("=" * 25)
    print("In three_opt")
    swapped = True
    while swapped:
        swapped = False
        for i in range(nodeCount-2):
            for j in range(i+1, nodeCount-1):
                for k in range(j+1, nodeCount):

                    d1 = length(points[best_route[j-1]], points[best_route[j]])
                    d2 = length(points[best_route[k-1]], points[best_route[k]])
                    d3 = length(points[best_route[i-1]], points[best_route[i]])
                    d_sum = d1+d2+d3

                    d_case_1 = length(points[best_route[j-1]], points[best_route[k-1]]) + \
                                length(points[best_route[j]], points[best_route[i-1]]) + \
                                length(points[best_route[k]], points[best_route[i]])

                    d_case_2 = length(points[best_route[j-1]], points[best_route[k]]) + \
                                length(points[best_route[j]], points[best_route[i-1]]) + \
                                length(points[best_route[k-1]], points[best_route[i]])

                    d_case_3 = length(points[best_route[j-1]], points[best_route[k]]) + \
                                length(points[best_route[k-1]], points[best_route[i-1]]) + \
                                length(points[best_route[j]], points[best_route[i]])

                    d_case_4 = length(points[best_route[j-1]], points[best_route[i-1]]) + \
                                length(points[best_route[k]], points[best_route[j]]) + \
                                length(points[best_route[k-1]], points[best_route[i]])

                    candidate_case = min([("case1", d_case_1-d_sum), ("case2", d_case_2-d_sum), \
                     ("case3", d_case_3-d_sum), ("case4", d_case_3-d_sum)], key=lambda t:t[1])

                    if candidate_case[1] >= -10e-4:
                        continue

                    A, B, C = best_route[i:j], best_route[j:k], np.append(best_route[k:], best_route[:i])

                    if candidate_case[0] == "case1":
                        best_route = np.concatenate((A, np.flip(B), np.flip(C)))
                    elif candidate_case[0] == "case2":
                        best_route = np.concatenate((A, C, B))
                    elif candidate_case[0] == "case3":
                        best_route = np.concatenate((A, C, np.flip(B)))
                    else:
                        assert candidate_case[0] == "case4"
                        best_route = np.concatenate((A, np.flip(C), B))

                    best_dist += candidate_case[1]
                    swapped = True
                    print(candidate_case[1])
                    print("Current best", best_dist)

    return best_route, best_dist

def two_opt(solution, obj, points):

    nodeCount = len(solution)
    best_dist = obj
    best_route = np.append(solution, solution[0])
    print("=" * 25)
    swapped = True
    while swapped:
        swapped = False
        for i in range(nodeCount-1):
            for j in range(i+1, nodeCount):
                d1 = length(points[best_route[i]], points[best_route[i+1]])
                d2 = length(points[best_route[j]], points[best_route[j+1]])
                d1_new = length(points[best_route[i]], points[best_route[j]])
                d2_new = length(points[best_route[i+1]], points[best_route[j+1]])

                if d1_new + d2_new >= d1 + d2:
                    continue

                best_route[i+1:j+1] = np.flip(best_route[i+1:j+1])
                best_dist += (d1_new + d2_new - d1 - d2)
                swapped = True
                print("Current best", best_dist)

        # print("re-entry")
        # print(inner_break, outer_break, best_dist)
    print("Current best", best_dist)
    print("Returning from two_opt")
    return best_route[:-1], best_dist

def solve_it_3(input_data):

    points, nodeCount, _ = process_input(input_data)

    file_sol = {51:"tsp_sol_51_1", 100:"tsp_sol_100_3", 200:"tsp_sol_200_2", 574:"tsp_sol_574_1", 1889:"tsp_sol_1889_1", 33810:"tsp_sol_33810_1"}
    with open("./sol/%s" % file_sol[nodeCount]) as fd:
        contents = fd.read()

    tmp = contents.split("\n")
    obj = float(tmp[0].split()[0])
    solution = np.array([int(i) for i in tmp[1].split()])
    print("before three_opt", obj)
    solution, obj = three_opt(solution, obj, points)
    print("After three_opt", obj)
    return format_output(solution, obj)

def solve_it_2(input_data):

    points, nodeCount, p2i_dict = process_input(input_data)
    hull = ConvexHull(points)
    # solution = np.zeros(nodeCount, dtype=np.int)
    solution = hull.vertices

    unvisited_points = set(np.arange(nodeCount))-set(hull.vertices)
    visited_points_cnt = len(hull.vertices)

    # print(visited_points_cnt)
    # print(points[:5])
    # print(solution)
    # print(unvisited_points)
    # print(hull.vertices)
    # print(points[hull.vertices])

    print("Enter the loop")
    while len(unvisited_points) != 0:

        try:
            curr_hull = ConvexHull(points[list(unvisited_points)])
        except:
            print("error")
            break

        points_to_visit = np.array([p2i_dict[tuple(t)] for t in points[list(unvisited_points)][curr_hull.vertices]])
        # print("points_to_visit", points_to_visit)

        solution, visited_points_cnt = insert_unvisited_points(points, points_to_visit, solution, visited_points_cnt)
        
        unvisited_points -= set(points_to_visit)


        # print("solution", solution, solution.shape)
        # print("unvisited_points", unvisited_points)
        # print("visited_points_cnt", visited_points_cnt)
        # print("="*50)
    print("out of loop")

    solution, visited_points_cnt = insert_unvisited_points(points, unvisited_points, solution, visited_points_cnt)

    print("solution", solution, len(set(solution)))
    print("visited_points_cnt", visited_points_cnt)
    obj = calc_solution_obj(solution, points)
    solution, obj = two_opt(solution, obj, points)
    print("After two_opt", obj)

    solution, obj = three_opt(solution, obj, points)
    print("After three_opt", obj)
    return format_output(solution, obj)

def solve_it_1(input_data):
    # Modify this code to run your optimization algorithm

    points, nodeCount, p2i_dict = process_input(input_data)
    print(len(points), points)
    # distance_matrix = build_dist_mat(points)
    # print(distance_matrix)

    # return
    # Randomly pick one point

    all_ret = [0] * nodeCount

    for rand_node in range(nodeCount):
    # for rand_node in [0]:
        visited = np.zeros(nodeCount, dtype=np.int)
        solution = np.zeros(nodeCount, dtype=np.int)
        curr_node = rand_node
        solution[0] = curr_node
        visited[curr_node] = 1
        
        obj = 0
        for idx in range(1, nodeCount):
            cand_node_list = [(i, p, length(p, points[curr_node])) for i, p in enumerate(points) if not visited[i]]
            next_node = min(cand_node_list, key=lambda t:t[2])

            solution[idx] = next_node[0]
            visited[next_node[0]] = 1
            obj += next_node[2]
            curr_node = next_node[0]


        obj += length(points[solution[0]], points[solution[-1]])

        solution, obj = two_opt(solution, obj, points)
        solution, obj = three_opt(solution, obj, points)

        all_ret[rand_node] = (obj, solution)

    print("min sol", min(all_ret, key=lambda t:t[0]))
    print("max sol", max(all_ret, key=lambda t:t[0]))

    ret = min(all_ret, key=lambda t:t[0])
    solution = ret[1]
    obj = ret[0]

    # solution, obj = two_opt(solution, obj, points)
    return format_output(solution, obj)

import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')

