# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@credits: "Thomas Runkler, and Anne Laurent,"
@license: "MIT"
@version: "2.0"
@email: "owuordickson@gmail.com"
@created: "26 July 2021"
@modified: "07 September 2021"


Breath-First Search for gradual patterns using Pure Local Search (PLS-GRAD).
PLS is used to learn gradual pattern candidates.

Adopted from: https://machinelearningmastery.com/iterated-local-search-from-scratch-in-python/

CHANGES:
1. Used rank order search space

"""

import random
import numpy as np
from bayes_opt import BayesianOptimization
from ypstruct import structure
import so4gp as sgp

from .shared.gp import GI, validate_gp, is_duplicate, check_anti_monotony
from .shared.dataset import Dataset
from .shared.search_spaces import Bitmap, Numeric


class LS_Numeric:

    # hill climbing local search algorithm
    @staticmethod
    def run(data_src, min_supp, max_iteration, step_size, nvar):
        max_iteration = int(max_iteration)
        nvar = int(nvar)

        # Prepare data set
        d_set = Dataset(data_src, min_supp)
        d_set.init_gp_attributes()
        attr_keys = [GI(x[0], x[1].decode()).as_string() for x in d_set.valid_bins[:, 0]]

        if d_set.no_bins:
            return []

        # Parameters
        it_count = 0
        var_min = 0
        var_max = int(''.join(['1'] * len(attr_keys)), 2)
        eval_count = 0

        # Empty Individual Template
        best_sol = structure()
        candidate = structure()
        # best_sol.position = None
        # best_sol.cost = float('inf')

        # INITIALIZE
        # best_sol.position = np.random.uniform(var_min, var_max, nvar)
        # generate an initial point
        best_sol.position = None
        # candidate.position = None
        if best_sol.position is None:
            best_sol.position = int(np.random.uniform(var_min, var_max, nvar)[0])
        # evaluate the initial point
        Numeric.apply_bound(best_sol, var_min, var_max)
        best_sol.cost = Numeric.cost_func(best_sol.position, attr_keys, d_set)

        # Best Cost of Iteration
        best_costs = np.empty(max_iteration)
        best_patterns = []
        str_iter = ''
        str_eval = ''

        invalid_count = 0
        all_encodings = []

        repeated = 0
        # run the hill climb
        while it_count < max_iteration:
            # while eval_count < max_evaluations:
            # take a step
            candidate.position = None
            if candidate.position is None:
                candidate.position = int(best_sol.position + (random.randrange(var_min, var_max) * step_size))
            Numeric.apply_bound(candidate, var_min, var_max)
            candidate.cost = Numeric.cost_func(candidate.position, attr_keys, d_set)

            if candidate.cost == 1:
                invalid_count += 1
            if candidate.cost < best_sol.cost:
                best_sol = candidate.deepcopy()
            eval_count += 1
            str_eval += "{}: {} \n".format(eval_count, best_sol.cost)
            all_encodings.append(candidate.position)

            best_gp = validate_gp(d_set, Numeric.decode_gp(attr_keys, best_sol.position))
            is_present = is_duplicate(best_gp, best_patterns)
            is_sub = check_anti_monotony(best_patterns, best_gp, subset=True)
            if is_present or is_sub:
                repeated += 1
            else:
                if best_gp.support >= min_supp:
                    best_patterns.append(best_gp)

            try:
                # Show Iteration Information
                # Store Best Cost
                best_costs[it_count] = best_sol.cost
                str_iter += "{}: {} \n".format(it_count, best_sol.cost)
            except IndexError:
                pass
            it_count += 1

        # Parameter Tuning - Output
        if data_src == 0.0:
            return 1/best_sol.cost

        # Output
        out = structure()
        out.best_sol = best_sol
        out.best_costs = best_costs
        out.best_patterns = best_patterns
        out.invalid_pattern_count = invalid_count
        out.total_candidates = all_encodings
        out.str_iterations = str_iter
        out.iteration_count = it_count
        out.max_iteration = max_iteration
        out.str_evaluations = str_eval
        out.cost_evaluations = eval_count
        out.step_size = step_size
        out.titles = d_set.titles
        out.col_count = d_set.col_count
        out.row_count = d_set.row_count
        return out

    @staticmethod
    def execute(f_path, min_supp, cores, max_iteration, step_size, nvar, visuals):
        try:
            if cores > 1:
                num_cores = cores
            else:
                num_cores = sgp.get_num_cores()

            out = LS_Numeric.run(f_path, min_supp, max_iteration, step_size, nvar)
            list_gp = out.best_patterns

            wr_line = "Algorithm: PLS-GRAANK (v2.0)\n"
            wr_line += "Search Space: Numeric\n"
            wr_line += "No. of (dataset) attributes: " + str(out.col_count) + '\n'
            wr_line += "No. of (dataset) tuples: " + str(out.row_count) + '\n'
            wr_line += "Step size: " + str(out.step_size) + '\n'
            wr_line += "N-var: " + str(nvar) + '\n'

            wr_line += "Number of iterations: " + str(out.iteration_count) + '\n'
            wr_line += "Number of cost evaluations: " + str(out.cost_evaluations) + '\n'
            wr_line += "Candidates: " + str(out.total_candidates) + '\n'

            wr_line += "Minimum support: " + str(min_supp) + '\n'
            wr_line += "Number of cores: " + str(num_cores) + '\n'
            wr_line += "Number of patterns: " + str(len(list_gp)) + '\n'
            wr_line += "Number of invalid patterns: " + str(out.invalid_pattern_count) + '\n\n'

            for txt in out.titles:
                try:
                    wr_line += (str(txt.key) + '. ' + str(txt.value.decode()) + '\n')
                except AttributeError:
                    wr_line += (str(txt[0]) + '. ' + str(txt[1].decode()) + '\n')

            wr_line += str("\nFile: " + f_path + '\n')
            wr_line += str("\nPattern : Support" + '\n')

            for gp in list_gp:
                wr_line += (str(gp.to_string()) + ' : ' + str(round(gp.support, 3)) + '\n')

            if visuals[1]:
                wr_line += '\n\n' + "Evaluation: Cost" + '\n'
                wr_line += out.str_evaluations
            if visuals[2]:
                wr_line += '\n\n' + "Iteration: Best Cost" + '\n'
                wr_line += out.str_iterations
            return wr_line
        except ArithmeticError as error:
            wr_line = "Failed: " + str(error)
            print(error)
            return wr_line


class LS_Bitmap:

    def __int__(self):
        pass

    @staticmethod
    def run(f_path, min_supp, max_iteration, step_size, nvar):
        # Prepare data set
        d_set = Dataset(f_path, min_supp)
        d_set.init_gp_attributes()
        attr_keys = [GI(x[0], x[1].decode()).as_string() for x in d_set.valid_bins[:, 0]]
        attr_keys_spl = [attr_keys[x:x + 2] for x in range(0, len(attr_keys), 2)]

        if d_set.no_bins:
            return []

        # Parameters
        it_count = 0
        eval_count = 0

        # Empty Individual Template
        best_sol = structure()
        candidate = structure()
        # best_sol.gene = None
        # best_sol.cost = float('inf')

        # generate an initial point
        best_sol.position = None
        best_sol.position = Bitmap.build_gp_gene(attr_keys_spl)
        best_sol.cost = Bitmap.cost_func(best_sol.position, attr_keys_spl, d_set)

        # Best Cost of Iteration
        best_costs = np.empty(max_iteration)
        best_patterns = []
        str_iter = ''
        str_eval = ''

        invalid_count = 0
        all_encodings = []

        repeated = 0
        # run the hill climb
        while it_count < max_iteration:
            # while eval_count < max_evaluations:
            # take a step
            candidate.position = None
            if candidate.position is None:
                alpha = np.random.uniform(-step_size, 1 + step_size, best_sol.position.shape[1])
                candidate.position = alpha * best_sol.position
            candidate.cost = Bitmap.cost_func(candidate.position, attr_keys_spl, d_set)
            if candidate.cost == 1:
                invalid_count += 1

            if candidate.cost < best_sol.cost:
                best_sol = candidate.deepcopy()
            eval_count += 1
            str_eval += "{}: {} \n".format(eval_count, best_sol.cost)
            all_encodings.append(Bitmap.decode_encoding(candidate.position))

            best_gp = validate_gp(d_set, Bitmap.decode_gp(attr_keys_spl, best_sol.position))
            is_present = is_duplicate(best_gp, best_patterns)
            is_sub = check_anti_monotony(best_patterns, best_gp, subset=True)
            if is_present or is_sub:
                repeated += 1
            else:
                if best_gp.support >= min_supp:
                    best_patterns.append(best_gp)

            try:
                # Show Iteration Information
                # Store Best Cost
                best_costs[it_count] = best_sol.cost
                str_iter += "{}: {} \n".format(it_count, best_sol.cost)
            except IndexError:
                pass
            it_count += 1

            # Output
        out = structure()
        out.best_sol = best_sol
        out.best_costs = best_costs
        out.best_patterns = best_patterns
        out.invalid_pattern_count = invalid_count
        out.total_candidates = all_encodings
        out.str_iterations = str_iter
        out.iteration_count = it_count
        out.max_iteration = max_iteration
        out.str_evaluations = str_eval
        out.cost_evaluations = eval_count
        out.step_size = step_size
        out.titles = d_set.titles
        out.col_count = d_set.col_count
        out.row_count = d_set.row_count
        return out

    @staticmethod
    def execute(f_path, min_supp, cores, max_iteration, step_size, nvar, visuals):
        try:
            if cores > 1:
                num_cores = cores
            else:
                num_cores = sgp.get_num_cores()

            out = LS_Bitmap.run(f_path, min_supp, max_iteration, step_size, nvar)
            list_gp = out.best_patterns

            wr_line = "Algorithm: PLS-GRAANK (v1.0)\n"
            wr_line += "Search Space: Bitmap\n"
            wr_line += "No. of (dataset) attributes: " + str(out.col_count) + '\n'
            wr_line += "No. of (dataset) tuples: " + str(out.row_count) + '\n'
            wr_line += "Step size: " + str(out.step_size) + '\n'
            wr_line += "N-var: " + str(nvar) + '\n'

            wr_line += "Number of iterations: " + str(out.iteration_count) + '\n'
            wr_line += "Number of cost evaluations: " + str(out.cost_evaluations) + '\n'
            wr_line += "Candidates: " + str(out.total_candidates) + '\n'

            wr_line += "Minimum support: " + str(min_supp) + '\n'
            wr_line += "Number of cores: " + str(num_cores) + '\n'
            wr_line += "Number of patterns: " + str(len(list_gp)) + '\n'
            wr_line += "Number of invalid patterns: " + str(out.invalid_pattern_count) + '\n\n'

            for txt in out.titles:
                try:
                    wr_line += (str(txt.key) + '. ' + str(txt.value.decode()) + '\n')
                except AttributeError:
                    wr_line += (str(txt[0]) + '. ' + str(txt[1].decode()) + '\n')

            wr_line += str("\nFile: " + f_path + '\n')
            wr_line += str("\nPattern : Support" + '\n')

            for gp in list_gp:
                wr_line += (str(gp.to_string()) + ' : ' + str(round(gp.support, 3)) + '\n')

            if visuals[1]:
                wr_line += '\n\n' + "Evaluation: Cost" + '\n'
                wr_line += out.str_evaluations
            if visuals[2]:
                wr_line += '\n\n' + "Iteration: Best Cost" + '\n'
                wr_line += out.str_iterations
            return wr_line
        except ArithmeticError as error:
            wr_line = "Failed: " + str(error)
            print(error)
            return wr_line


def parameter_tuning():
    pbounds = {'data_src': (0, 0), 'min_supp': (0.5, 0.5), 'max_iteration': (1, 10), 'step_size': (0.1, 1),
               'nvar': (1, 1)}

    optimizer = BayesianOptimization(
        f=LS_Numeric.run,
        pbounds=pbounds,
        random_state=1,
    )

    optimizer.maximize(
        init_points=10,
        n_iter=0,
    )
    return optimizer.max
