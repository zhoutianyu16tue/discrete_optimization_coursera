#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
import math
import numpy as np
from time import sleep
from itertools import combinations

Point = namedtuple("Point", ['x', 'y'])
Facility = namedtuple("Facility", ['index', 'setup_cost', 'capacity', 'location'])
Customer = namedtuple("Customer", ['index', 'demand', 'location'])

def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def process_input(input_data):
        # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    facility_count = int(parts[0])
    customer_count = int(parts[1])
    
    facilities = []
    for i in range(1, facility_count+1):
        parts = lines[i].split()
        facilities.append(Facility(i-1, float(parts[0]), int(parts[1]), Point(float(parts[2]), float(parts[3])) ))

    customers = []
    for i in range(facility_count+1, facility_count+1+customer_count):
        parts = lines[i].split()
        customers.append(Customer(i-1-facility_count, int(parts[0]), Point(float(parts[1]), float(parts[2]))))

    no_facilities, no_customers = len(facilities), len(customers)
    cust_facility_dist = np.zeros((no_customers, no_facilities))

    for customer in customers:
        for facility in facilities:
            cust_facility_dist[customer.index, facility.index] = length(customer.location, facility.location)

    return facilities, customers, cust_facility_dist

def format_output(solution, obj, opt=0):
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))
    return output_data

def solve_it_1(customers, facilities, customer_facility_dist):
    solution = [-1]*len(customers)
    capacity_remaining = [f.capacity for f in facilities]

    facility_index = 0
    for customer in customers:
        if capacity_remaining[facility_index] >= customer.demand:
            solution[customer.index] = facility_index
            capacity_remaining[facility_index] -= customer.demand
        else:
            facility_index += 1
            assert capacity_remaining[facility_index] >= customer.demand
            solution[customer.index] = facility_index
            capacity_remaining[facility_index] -= customer.demand

    used = [0]*len(facilities)
    for facility_index in solution:
        used[facility_index] = 1

    # calculate the cost of the solution
    obj = sum([f.setup_cost*used[f.index] for f in facilities])
    for customer in customers:
        obj += length(customer.location, facilities[solution[customer.index]].location)

    return solution, obj

def solve_it_2(customers, facilities, customer_facility_dist):
    
    no_customers = customer_facility_dist.shape[0]
    no_facilities = customer_facility_dist.shape[1]
    solution = np.zeros(no_customers, dtype=np.int)
    obj = 0
    selected_facilities = np.zeros(no_facilities, dtype=np.int)
    facility_costs = np.array([f.setup_cost for f in facilities])
    facility_capacities = np.array([f.capacity for f in facilities])
    facility_cap_remain = np.array([f.capacity for f in facilities])

    for customer in customers:
        facility_index = np.argmin(customer_facility_dist[customer.index])

        # The closest facility has capacity
        if facility_cap_remain[facility_index] <= customer.demand:

            faci_indices = np.argsort(customer_facility_dist[customer.index])
            assert faci_indices[0] == facility_index

            for fi in faci_indices[1:]:
                if facility_cap_remain[fi] >= customer.demand:
                    facility_index = fi
                    break

        selected_facilities[facility_index] = 1
        solution[customer.index] = facility_index
        obj += customer_facility_dist[customer.index, facility_index]
        facility_cap_remain[facility_index] -= customer.demand

    obj += selected_facilities.dot(facility_costs)

    facility_utilities = 1 - facility_cap_remain/ facility_capacities
    # print("Facility capacity utility")
    # print(facility_utilities)
    # print(np.argsort(facility_utilities))
    # print()

    # Start Local search
    improved = True
    while improved:
        improved = False
        # print("beginning of the loop")
        # print(facility_utilities)
        for fi in np.argsort(facility_utilities):
            if facility_utilities[fi] == 0:
                continue
            # print("current facility_utilities", facility_utilities)
            # print("current solution", solution, fi)
            customers_to_relocate = np.argwhere(solution == fi).flatten()

            if len(customers_to_relocate) == 0:
                continue
            # print("Current Fi", fi)
            # print("customers_to_relocate", customers_to_relocate)
            relocation_records = [0] * len(customers_to_relocate)

            facility_capacity_potential_decrease = np.zeros(no_facilities)

            for ci, c in enumerate(customers_to_relocate):
                faci_indices = np.argsort(customer_facility_dist[c])
                tmp = np.argwhere(faci_indices==fi).flatten()
                assert len(tmp) == 1
                curr_faci_idx = tmp[0]

                relocated = False

                for i in range(curr_faci_idx+1, no_facilities):
                    if not selected_facilities[faci_indices[i]] or (facility_cap_remain[faci_indices[i]] - facility_capacity_potential_decrease[faci_indices[i]]) < customers[c].demand:
                        continue

                    relocated = True
                    new_facility = faci_indices[i]
                    facility_capacity_potential_decrease[new_facility] += customers[c].demand
                    break

                if not relocated:
                    break
                
                cost_increase = customer_facility_dist[c, new_facility] - customer_facility_dist[c, fi]
                relocation_records[ci] = (relocated, cost_increase, c, new_facility, customers[c].demand)

            if not relocated:
                continue

            # print()
            
            max_cost_decrease = facility_costs[fi]
            min_cost_increase = sum([rr[1] for rr in relocation_records])
            
            if min_cost_increase >= max_cost_decrease:
                continue

            # print(facility_capacity_potential_decrease)
            # print("max_cost_decrease, min_cost_increase", max_cost_decrease, min_cost_increase)
            # print("facility_cap_remain before update", facility_cap_remain)
            obj -= (max_cost_decrease - min_cost_increase)
            for rr in relocation_records:
                c, f = rr[2], rr[3]
                solution[c] = f
                assert facility_cap_remain[f] >= customers[c].demand
                facility_cap_remain[f] -= customers[c].demand

            selected_facilities[fi] = 0
            improved = True
            facility_utilities = 1 - facility_cap_remain/ facility_capacities
            facility_utilities[fi] = 0
            # print("=" * 25)
            # print("current obj, fi", obj, fi)
            # print(facility_utilities)
            # print("relocation_records", relocation_records)
            # # print("=" * 25)
            # print("facility_cap_remain after update", facility_cap_remain)
            # sleep(1)
            break
    # print("facility_utilities")
    # print(1 - facility_cap_remain/ facility_capacities)
    # print("="*40)

    def calc_mvt_mvf_cost(f_move_to, f_move_from):
        customers_to_relocate = np.argwhere(solution==f_move_from).flatten()
        if len(customers_to_relocate) > 0:
            cost_decrease = np.sum(customer_facility_dist[customers_to_relocate, f_move_from])
            cost_decrease += facility_costs[f_move_from]

            cost_increase = np.sum(customer_facility_dist[customers_to_relocate, f_move_to])
            if not selected_facilities[f_move_to]:
                cost_increase += facility_costs[f_move_to]

            if cost_decrease > cost_increase:
                print("facility %d has enough capacity" % f_move_to, f_move_from, cost_decrease, cost_increase)
                print(facilities[f_move_to].location, facilities[f_move_from].location)
                print(facility_cap_usage[f_move_to], facility_cap_usage[f_move_from])
                print(facilities[f_move_to].capacity, facilities[f_move_from].capacity)
                print()
                return cost_decrease - cost_increase

        return  0

    facility_cap_usage = facility_capacities - facility_cap_remain
    total_decrease = 0
    for f1, f2 in combinations(np.arange(no_facilities), 2):
        if facility_cap_usage[f1] + facility_cap_usage[f2] <= facility_capacities[f1]:
            f_move_to, f_move_from = f1, f2
            total_decrease += calc_mvt_mvf_cost(f_move_to, f_move_from)

        if facility_cap_usage[f1] + facility_cap_usage[f2] <= facility_capacities[f2]:
            f_move_to, f_move_from = f2, f1
            total_decrease += calc_mvt_mvf_cost(f_move_to, f_move_from)

    print(obj, total_decrease, obj-total_decrease)
        # print("=" * 30)
    return solution, obj

def solve_it(input_data):
    # Modify this code to run your optimization algorithm
    facilities, customers, customer_facility_dist = process_input(input_data)

    print(customer_facility_dist.shape)
    # build a trivial solution
    # pack the facilities one by one until all the customers are served

    # solution, obj = solve_it_1(customers, facilities, customer_facility_dist)
    # print("Naive", obj)
    solution, obj = solve_it_2(customers, facilities, customer_facility_dist)
    print("Simple", obj)
    # return format_output(solution, obj)

import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/fl_16_2)')

