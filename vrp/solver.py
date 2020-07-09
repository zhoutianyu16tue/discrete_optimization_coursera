#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from collections import namedtuple
import numpy as np

Customer = namedtuple("Customer", ['index', 'demand', 'x', 'y'])

def length(customer1, customer2):
    return math.sqrt((customer1.x - customer2.x)**2 + (customer1.y - customer2.y)**2)

def process_input(input_data):
        # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    customer_count = int(parts[0])
    vehicle_count = int(parts[1])
    vehicle_capacity = int(parts[2])
    
    customers = []
    for i in range(1, customer_count+1):
        line = lines[i]
        parts = line.split()
        customers.append(Customer(i-1, int(parts[0]), float(parts[1]), float(parts[2])))

    #the depot is always the first customer in the input
    depot = customers[0]
    return depot, customers, customer_count, vehicle_count, vehicle_capacity

def length2(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calc_route_cost(route, points):

    assert len(route) == len(points)
    cost = 0
    for i in range(len(route)-1):
        cost += length2(points[route[i]], points[route[i+1]])

    cost += length2(points[route[-1]], points[route[0]])

    return cost

def three_opt(solution, obj, points):

    nodeCount = len(solution)
    best_dist = obj
    best_route = solution
    swapped = True
    while swapped:
        swapped = False
        for i in range(nodeCount-2):
            for j in range(i+1, nodeCount-1):
                for k in range(j+1, nodeCount):

                    d1 = length2(points[best_route[j-1]], points[best_route[j]])
                    d2 = length2(points[best_route[k-1]], points[best_route[k]])
                    d3 = length2(points[best_route[i-1]], points[best_route[i]])
                    d_sum = d1+d2+d3

                    d_case_1 = length2(points[best_route[j-1]], points[best_route[k-1]]) + \
                                length2(points[best_route[j]], points[best_route[i-1]]) + \
                                length2(points[best_route[k]], points[best_route[i]])

                    d_case_2 = length2(points[best_route[j-1]], points[best_route[k]]) + \
                                length2(points[best_route[j]], points[best_route[i-1]]) + \
                                length2(points[best_route[k-1]], points[best_route[i]])

                    d_case_3 = length2(points[best_route[j-1]], points[best_route[k]]) + \
                                length2(points[best_route[k-1]], points[best_route[i-1]]) + \
                                length2(points[best_route[j]], points[best_route[i]])

                    d_case_4 = length2(points[best_route[j-1]], points[best_route[i-1]]) + \
                                length2(points[best_route[k]], points[best_route[j]]) + \
                                length2(points[best_route[k-1]], points[best_route[i]])

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

    return best_route, best_dist

def two_opt(customers):

    nodeCount = len(customers)
    solution = [c.index for c in customers]
    points = {c.index:(c.x, c.y) for c in customers}

    best_dist = calc_route_cost(solution, points)
    best_route = np.append(solution, solution[0])
    # print("=" * 25)
    # print("Before opt", best_dist)
    swapped = True
    while swapped:
        swapped = False
        for i in range(nodeCount-1):
            for j in range(i+1, nodeCount):
                d1 = length2(points[best_route[i]], points[best_route[i+1]])
                d2 = length2(points[best_route[j]], points[best_route[j+1]])
                d1_new = length2(points[best_route[i]], points[best_route[j]])
                d2_new = length2(points[best_route[i+1]], points[best_route[j+1]])

                if d1_new + d2_new >= d1 + d2:
                    continue

                best_route[i+1:j+1] = np.flip(best_route[i+1:j+1])
                best_dist += (d1_new + d2_new - d1 - d2)
                swapped = True
                # print("Current best", best_dist)

        # print("re-entry")
        # print(inner_break, outer_break, best_dist)
    # print("Returning from two_opt")
    # return best_route[:-1], best_dist
    return three_opt(best_route[:-1], best_dist, points)

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    depot, customers, customer_count, vehicle_count, vehicle_capacity = \
        process_input(input_data)


    # build a trivial solution
    # assign customers to vehicles starting by the largest customer demands
    vehicle_tours = []
    
    remaining_customers = set(customers)
    remaining_customers.remove(depot)
    
    for v in range(0, vehicle_count):
        # print "Start Vehicle: ",v
        vehicle_tours.append([])
        capacity_remaining = vehicle_capacity
        curr_pos = depot

        while capacity_remaining >= 0 and len(remaining_customers) > 0:
            customer_by_dist = sorted([(length(curr_pos, c), c) for c in remaining_customers], key=lambda t:t[0])
            # print(customer_by_dist)
            # print("remaining_customers", len(remaining_customers))
            # print("capacity_remaining", capacity_remaining)
            found = False
            for _, customer in customer_by_dist:
                if customer.demand <= capacity_remaining:
                    found = True
                    break

            if not found:
                break
            curr_pos = customer
            capacity_remaining -= customer.demand
            remaining_customers -= set([customer])
            vehicle_tours[v].append(customer)

        # while sum([capacity_remaining >= customer.demand for customer in remaining_customers]) > 0:
        #     used = set()
        #     order = sorted(remaining_customers, key=lambda customer: -customer.demand*customer_count + customer.index)
        #     print(order)
        #     for customer in order:
        #         if capacity_remaining >= customer.demand:
        #             capacity_remaining -= customer.demand
        #             vehicle_tours[v].append(customer)
        #             # print '   add', ci, capacity_remaining
        #             used.add(customer)
        #     remaining_customers -= used

    # checks that the number of customers served is correct
    print(len(vehicle_tours))
    print(sum([len(v) for v in vehicle_tours]))
    print(customer_count)
    # assert sum([len(v) for v in vehicle_tours]) == len(customers) - 1

    # prepare the solution in the specified output format
    obj_opt = 0
    outputData = ""
    for v in range(0, vehicle_count):

        curr_tour = vehicle_tours[v]
        if len(curr_tour) == 0:
            print("un-used vehicle")
            outputData += "0 0\n"
            continue

        r1, c1 = two_opt([depot]+curr_tour)
        tour_start = np.argwhere(r1==0).flatten()[0]
        r1 = np.append(r1[:tour_start], r1[tour_start:])
        # print("route", v, c1)
        obj_opt += c1
        outputData += "0" + ' ' + ' '.join([str(i) for i in r1[1:]]) + ' ' + "0" + '\n'

    outputData = '%.2f' % obj_opt + ' ' + str(0) + '\n' + outputData
    return outputData


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:

        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/vrp_5_4_1)')

