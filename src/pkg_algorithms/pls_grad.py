# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@credits: "Thomas Runkler, and Anne Laurent,"
@license: "MIT"
@version: "1.0"
@email: "owuordickson@gmail.com"
@created: "26 July 2021"

Breath-First Search for gradual patterns using Pure Local Search (PLS-GRAD).
PLS is used to learn gradual pattern candidates.

CHANGES:
1.

"""

import random
import numpy as np
from ypstruct import structure
import matplotlib.pyplot as plt

from .shared.gp import GI, GP
from .shared.dataset_bfs import Dataset
from .shared.profile import Profile
from .shared import config as cfg


# hill climbing local search algorithm
def run_hill_climbing(f_path, min_supp, max_iteration=cfg.MAX_ITERATIONS, step_size=cfg.STEP_SIZE):
    # Prepare data set
    d_set = Dataset(f_path, min_supp)
    d_set.init_gp_attributes()
    attr_keys = [GI(x[0], x[1].decode()).as_string() for x in d_set.valid_bins[:, 0]]

    if d_set.no_bins:
        return []

    # Parameters
    it_count = 0
    var_min = 0
    var_max = int(''.join(['1'] * len(attr_keys)), 2)
    nvar = cfg.N_VAR

    # Empty Individual Template
    best_sol = structure()
    candidate = structure()
    # best_sol.position = None
    # best_sol.cost = float('inf')

    # INITIALIZE
    # best_sol.position = np.random.uniform(var_min, var_max, nvar)

    # Best Cost of Iteration
    best_costs = np.empty(max_iteration)
    best_patterns = []
    str_plt = ''
    repeated = 0

    # generate an initial point
    best_sol.position = None
    # candidate.position = None
    while best_sol.position is None or not apply_bound(best_sol, var_min, var_max):
        best_sol.position = np.random.uniform(var_min, var_max, nvar)
    # evaluate the initial point
    best_sol.cost = cost_func(best_sol.position, attr_keys, d_set)
    # run the hill climb
    while it_count < max_iteration:
        # take a step
        candidate.position = None
        while candidate.position is None or not apply_bound(candidate, var_min, var_max):
            candidate.position = best_sol.position + (random.random() * step_size * best_sol.position)
            print(str(candidate.position) + '+ ' + str(random.random() * step_size) + '= ' + str(best_sol.position) )
        # evaluate candidate.position point
        candidate.cost = cost_func(candidate.position, attr_keys, d_set)
        # check if we should keep the new point
        if candidate.cost < best_sol.cost:
            # store the new point
            best_sol = candidate.deepcopy()
            # report progress
            print('>%d f(%s) = %.5f' % (it_count, best_sol.position, best_sol.cost))
        # it_count += 1

        best_gp = validate_gp(d_set, decode_gp(attr_keys, best_sol.position))
        is_present = is_duplicate(best_gp, best_patterns)
        is_sub = check_anti_monotony(best_patterns, best_gp, subset=True)
        if is_present or is_sub:
            repeated += 1
        else:
            if best_gp.support >= min_supp:
                best_patterns.append(best_gp)
            else:
                best_sol.cost = 1

        try:
            # Show Iteration Information
            # Store Best Cost
            best_costs[it_count] = best_sol.cost
            str_plt += "Iteration {}: Best Cost: {} \n".format(it_count, best_costs[it_count])
        except IndexError:
            pass
        it_count += 1

        # Output
    out = structure()
    out.best_sol = best_sol
    out.best_costs = best_costs
    out.best_patterns = best_patterns
    out.str_iterations = str_plt
    out.iteration_count = it_count
    out.max_iteration = max_iteration
    out.titles = d_set.titles
    out.col_count = d_set.col_count
    out.row_count = d_set.row_count

    return out


def cost_func(position, attr_keys, d_set):
    bin_sum = compute_bin_sum(d_set, decode_gp(attr_keys, position))
    if bin_sum > 0:
        cost = (1 / bin_sum)
    else:
        cost = 1
    return cost


def apply_bound(x, var_min, var_max):
    if x.position < var_min or x.position > var_max:
        return False
    return True


def decode_gp(attr_keys, position):
    temp_gp = GP()
    if position is None:
        return temp_gp

    bin_str = bin(int(position))[2:]
    bin_arr = np.array(list(bin_str), dtype=int)

    for i in range(bin_arr.size):
        gene_val = bin_arr[i]
        if gene_val == 1:
            gi = GI.parse_gi(attr_keys[i])
            if not temp_gp.contains_attr(gi):
                temp_gp.add_gradual_item(gi)
    return temp_gp


def validate_gp(d_set, pattern):

    # pattern = [('2', '+'), ('4', '+')]
    min_supp = d_set.thd_supp
    n = d_set.attr_size
    gen_pattern = GP()
    bin_arr = np.array([])

    for gi in pattern.gradual_items:
        arg = np.argwhere(np.isin(d_set.valid_bins[:, 0], gi.gradual_item))
        if len(arg) > 0:
            i = arg[0][0]
            valid_bin = d_set.valid_bins[i]
            if bin_arr.size <= 0:
                bin_arr = np.array([valid_bin[1], valid_bin[1]])
                gen_pattern.add_gradual_item(gi)
            else:
                bin_arr[1] = valid_bin[1].copy()
                temp_bin = np.multiply(bin_arr[0], bin_arr[1])
                supp = float(np.sum(temp_bin)) / float(n * (n - 1.0) / 2.0)
                if supp >= min_supp:
                    bin_arr[0] = temp_bin.copy()
                    gen_pattern.add_gradual_item(gi)
                    gen_pattern.set_support(supp)
    if len(gen_pattern.gradual_items) <= 1:
        return pattern
    else:
        return gen_pattern


def compute_bin_sum(d_set, pattern):
    temp_bin = np.array([])
    # bin_arr = np.array([])
    for gi in pattern.gradual_items:
        arg = np.argwhere(np.isin(d_set.valid_bins[:, 0], gi.gradual_item))
        if len(arg) > 0:
            i = arg[0][0]
            valid_bin = d_set.valid_bins[i]
            if temp_bin.size <= 0:
                temp_bin = valid_bin[1].copy()
                # bin_arr = np.array([valid_bin[1], valid_bin[1]])
                # gen_pattern.add_gradual_item(gi)
            else:
                temp_bin = np.multiply(temp_bin, valid_bin[1])
                # bin_arr[1] = valid_bin[1].copy()
                # temp_bin = np.multiply(bin_arr[0], bin_arr[1])
                # supp = float(np.sum(temp_bin)) / float(n * (n - 1.0) / 2.0)
                # if supp >= min_supp:
                # bin_arr[0] = temp_bin.copy()
                # gen_pattern.add_gradual_item(gi)
                # gen_pattern.set_support(supp)
    return np.sum(temp_bin)


def check_anti_monotony(lst_p, pattern, subset=True):
    result = False
    if subset:
        for pat in lst_p:
            result1 = set(pattern.get_pattern()).issubset(set(pat.get_pattern()))
            result2 = set(pattern.inv_pattern()).issubset(set(pat.get_pattern()))
            if result1 or result2:
                result = True
                break
    else:
        for pat in lst_p:
            result1 = set(pattern.get_pattern()).issuperset(set(pat.get_pattern()))
            result2 = set(pattern.inv_pattern()).issuperset(set(pat.get_pattern()))
            if result1 or result2:
                result = True
                break
    return result


def is_duplicate(pattern, lst_winners):
    for pat in lst_winners:
        if set(pattern.get_pattern()) == set(pat.get_pattern()) or \
                set(pattern.inv_pattern()) == set(pat.get_pattern()):
            return True
    return False


def init(f_path, min_supp, cores):
    try:
        if cores > 1:
            num_cores = cores
        else:
            num_cores = Profile.get_num_cores()

        out = run_hill_climbing(f_path, min_supp)
        list_gp = out.best_patterns

        # Results
        plt.plot(out.best_costs)
        plt.semilogy(out.best_costs)
        plt.xlim(0, out.max_it)
        plt.xlabel('Iterations')
        plt.ylabel('Best Cost')
        plt.title('Genetic Algorithm (GA)')
        plt.grid(True)
        plt.show()

        wr_line = "Algorithm: PLS-GRAANK (v1.0)\n"
        wr_line += "No. of (dataset) attributes: " + str(out.col_count) + '\n'
        wr_line += "No. of (dataset) tuples: " + str(out.row_count) + '\n'
        # wr_line += "Population size: " + str(out.n_pop) + '\n'
        # wr_line += "PC: " + str(out.pc) + '\n'

        wr_line += "Minimum support: " + str(min_supp) + '\n'
        wr_line += "Number of cores: " + str(num_cores) + '\n'
        wr_line += "Number of patterns: " + str(len(list_gp)) + '\n'
        wr_line += "Number of iterations: " + str(out.iteration_count) + '\n\n'

        for txt in out.titles:
            try:
                wr_line += (str(txt.key) + '. ' + str(txt.value.decode()) + '\n')
            except AttributeError:
                wr_line += (str(txt[0]) + '. ' + str(txt[1].decode()) + '\n')

        wr_line += str("\nFile: " + f_path + '\n')
        wr_line += str("\nPattern : Support" + '\n')

        for gp in list_gp:
            wr_line += (str(gp.to_string()) + ' : ' + str(gp.support) + '\n')

        wr_line += '\n\nIterations \n'
        wr_line += out.str_iterations
        return wr_line
    except ArithmeticError as error:
        wr_line = "Failed: " + str(error)
        print(error)
        return wr_line