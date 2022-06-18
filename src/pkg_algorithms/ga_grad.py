# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@credits: "Thomas Runkler, and Anne Laurent,"
@license: "MIT"
@version: "2.0"
@email: "owuordickson@gmail.com"
@created: "29 April 2021"
@modified: "07 September 2021"

Breath-First Search for gradual patterns using Genetic Algorithm (GA-GRAD).
GA is used to learn gradual pattern candidates.

CHANGES:
1. uses normal functions
2. updated cost function to use Binary Array of GPs
3. uses rank order search space

"""


import random
import numpy as np
from bayes_opt import BayesianOptimization
from ypstruct import structure
import so4gp as sgp

from .shared.gp import GI, validate_gp, is_duplicate, check_anti_monotony
from .shared.dataset import Dataset
from .shared.search_spaces import Bitmap, Numeric


class GA_Numeric:

    def __int__(self):
        pass

    @staticmethod
    def run(data_src, min_supp, max_iteration, n_pop, pc, gamma, mu, sigma):

        max_iteration = int(max_iteration)

        n_pop = int(n_pop)

        # Prepare data set
        d_set = Dataset(data_src, min_supp)
        d_set.init_gp_attributes()
        attr_keys = [GI(x[0], x[1].decode()).as_string() for x in d_set.valid_bins[:, 0]]

        if d_set.no_bins:
            return []

        # Parameters
        # pc: Proportion of children (if its 1, then nc == npop
        it_count = 0
        eval_count = 0
        var_min = 0
        var_max = int(''.join(['1'] * len(attr_keys)), 2)

        nc = int(np.round(pc * n_pop / 2) * 2)  # Number of children. np.round is used to get even number of children

        # Empty Individual Template
        empty_individual = structure()
        empty_individual.position = None
        empty_individual.cost = None

        # Initialize Population
        pop = empty_individual.repeat(n_pop)
        for i in range(n_pop):
            pop[i].position = random.randrange(var_min, var_max)
            pop[i].cost = 1  # cost_func(pop[i].position, attr_keys, d_set)
            # if pop[i].cost < best_sol.cost:
            #    best_sol = pop[i].deepcopy()

        # Best Solution Ever Found
        best_sol = empty_individual.deepcopy()
        best_sol.position = pop[0].position
        best_sol.cost = Numeric.cost_func(best_sol.position, attr_keys, d_set)

        # Best Cost of Iteration
        best_costs = np.empty(max_iteration)
        best_patterns = []
        str_iter = ''
        str_eval = ''

        invalid_count = 0  # TO BE REMOVED
        all_encodings = []  # TO BE REMOVED

        repeated = 0
        while it_count < max_iteration:
            # while eval_count < max_evaluations:
            # while repeated < 1:

            c_pop = []  # Children population
            for _ in range(nc // 2):
                # Select Parents
                q = np.random.permutation(n_pop)
                p1 = pop[q[0]]
                p2 = pop[q[1]]

                # a. Perform Crossover
                c1, c2 = GA_Numeric.crossover(p1, p2, gamma)

                # Apply Bound
                Numeric.apply_bound(c1, var_min, var_max)
                Numeric.apply_bound(c2, var_min, var_max)

                # Evaluate First Offspring
                c1.cost = Numeric.cost_func(c1.position, attr_keys, d_set)
                if c1.cost == 1:
                    invalid_count += 1
                if c1.cost < best_sol.cost:
                    best_sol = c1.deepcopy()
                eval_count += 1
                str_eval += "{}: {} \n".format(eval_count, best_sol.cost)

                # Evaluate Second Offspring
                c2.cost = Numeric.cost_func(c2.position, attr_keys, d_set)
                if c2.cost == 1:
                    invalid_count += 1
                if c2.cost < best_sol.cost:
                    best_sol = c2.deepcopy()
                eval_count += 1
                str_eval += "{}: {} \n".format(eval_count, best_sol.cost)

                all_encodings.append([c1.position, Numeric.check_validity(c1.cost)])
                all_encodings.append([c2.position, Numeric.check_validity(c2.cost)])

                # b. Perform Mutation
                c1 = GA_Numeric.mutate(c1, mu, sigma)
                c2 = GA_Numeric.mutate(c2, mu, sigma)

                # Apply Bound
                Numeric.apply_bound(c1, var_min, var_max)
                Numeric.apply_bound(c2, var_min, var_max)

                # Evaluate First Offspring
                c1.cost = Numeric.cost_func(c1.position, attr_keys, d_set)
                if c1.cost == 1:
                    invalid_count += 1
                if c1.cost < best_sol.cost:
                    best_sol = c1.deepcopy()
                eval_count += 1
                str_eval += "{}: {} \n".format(eval_count, best_sol.cost)

                # Evaluate Second Offspring
                c2.cost = Numeric.cost_func(c2.position, attr_keys, d_set)
                if c2.cost == 1:
                    invalid_count += 1
                if c2.cost < best_sol.cost:
                    best_sol = c2.deepcopy()
                eval_count += 1
                str_eval += "{}: {} \n".format(eval_count, best_sol.cost)

                # c. Add Offsprings to c_pop
                c_pop.append(c1)
                c_pop.append(c2)

                all_encodings.append([c1.position, Numeric.check_validity(c1.cost)])
                all_encodings.append([c2.position, Numeric.check_validity(c2.cost)])

            # Merge, Sort and Select
            pop += c_pop
            pop = sorted(pop, key=lambda x: x.cost)
            pop = pop[0:n_pop]

            best_gp = validate_gp(d_set, Numeric.decode_gp(attr_keys, best_sol.position))
            is_present = is_duplicate(best_gp, best_patterns)
            is_sub = check_anti_monotony(best_patterns, best_gp, subset=True)
            if is_present or is_sub:
                repeated += 1
            else:
                if best_gp.support >= min_supp:
                    best_patterns.append(best_gp)
                # else:
                #    best_sol.cost = 1

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
            return 1 / best_sol.cost

        # Output
        out = structure()
        out.pop = pop
        out.best_sol = best_sol
        out.best_costs = best_costs
        out.best_patterns = best_patterns
        out.invalid_pattern_count = invalid_count
        out.total_candidates = all_encodings
        out.str_iterations = str_iter
        out.str_evaluations = str_eval
        out.iteration_count = it_count
        out.max_iteration = max_iteration
        out.cost_evaluations = eval_count
        out.n_pop = n_pop
        out.pc = pc
        out.titles = d_set.titles
        out.col_count = d_set.col_count
        out.row_count = d_set.row_count
        return out

    @staticmethod
    def crossover(p1, p2, gamma=0.1):
        c1 = p1.deepcopy()

        c2 = p2.deepcopy()
        alpha = np.random.uniform(0, gamma, 1)[0]
        c1.position = int(alpha * p1.position + (1 - alpha) * p2.position)
        c2.position = int(alpha * p2.position + (1 - alpha) * p1.position)
        return c1, c2

    @staticmethod
    def mutate(x, mu, sigma):
        y = x.deepcopy()

        str_x = str(int(y.position))
        # flag = np.random.rand(*x.position.shape) <= mu
        # ind = np.argwhere(flag)
        # y.position[ind] += sigma*np.random.rand(*ind.shape)
        flag = np.random.rand(*(len(str_x),)) <= mu
        ind = np.argwhere(flag)
        str_y = "0"
        for i in ind:
            val = float(str_x[i[0]])
            val += sigma * np.random.uniform(0, 1, 1)
            if i[0] == 0:
                str_y = "".join(("", "{}".format(int(val)), str_x[1:]))
            else:
                str_y = "".join((str_x[:i[0] - 1], "{}".format(int(val)), str_x[i[0]:]))
            str_x = str_y
        y.position = int(str_y)
        return y

    @staticmethod
    def execute(f_path, min_supp, cores, max_iteration, n_pop, pc, gamma, mu, sigma, visuals):
        try:
            if cores > 1:
                num_cores = cores
            else:
                num_cores = sgp.get_num_cores()

            out = GA_Numeric.run(f_path, min_supp, max_iteration, n_pop, pc, gamma, mu, sigma)
            list_gp = out.best_patterns

            wr_line = "Algorithm: GA-GRAANK (v2.0)\n"
            wr_line += "Search Space: Numeric\n"
            wr_line += "No. of (dataset) attributes: " + str(out.col_count) + '\n'
            wr_line += "No. of (dataset) tuples: " + str(out.row_count) + '\n'
            wr_line += "Population size: " + str(out.n_pop) + '\n'
            wr_line += "PC: " + str(out.pc) + '\n'
            wr_line += "Gamma: " + str(gamma) + '\n'
            wr_line += "Mu: " + str(mu) + '\n'
            wr_line += "Sigma: " + str(sigma) + '\n'

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


class GA_Bitmap:

    def __int__(self):
        pass

    @staticmethod
    def run(f_path, min_supp, max_iteration, n_pop, pc, gamma, mu, sigma):
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
        nc = int(np.round(pc * n_pop / 2) * 2)  # Number of children. np.round is used to get even number of children

        # Empty Individual Template
        empty_individual = structure()
        empty_individual.gene = None
        empty_individual.cost = None

        # Initialize Population
        pop = empty_individual.repeat(n_pop)
        for i in range(n_pop):
            pop[i].gene = Bitmap.build_gp_gene(attr_keys_spl)
            pop[i].cost = 1
            # print(pop[i].gene)
            # print(pop[i].gene.shape)
            # print(attr_keys_spl)

        # Best Solution Ever Found
        best_sol = empty_individual.deepcopy()
        best_sol.gene = pop[0].gene
        best_sol.cost = Bitmap.cost_func(best_sol.gene, attr_keys_spl, d_set)

        # Best Cost of Iteration
        best_costs = np.empty(max_iteration)
        best_genes = []
        best_patterns = []
        str_iter = ''
        str_eval = ''

        invalid_count = 0
        all_encodings = []

        repeated = 0
        while it_count < max_iteration:
            # while eval_count < max_evaluations:
            # while repeated < 1:

            c_pop = []  # Children population
            for _ in range(nc // 2):
                # Select Parents
                q = np.random.permutation(n_pop)
                p1 = pop[q[0]]
                p2 = pop[q[1]]

                # a. Perform Crossover
                c1, c2 = GA_Bitmap.crossover(p1, p2, gamma)

                # Evaluate First Offspring
                c1.cost = Bitmap.cost_func(c1.gene, attr_keys_spl, d_set)
                if c1.cost == 1:
                    invalid_count += 1
                if c1.cost < best_sol.cost:
                    best_sol = c1.deepcopy()
                eval_count += 1
                str_eval += "{}: {} \n".format(eval_count, best_sol.cost)

                # Evaluate Second Offspring
                c2.cost = Bitmap.cost_func(c2.gene, attr_keys_spl, d_set)
                if c2.cost == 1:
                    invalid_count += 1
                if c2.cost < best_sol.cost:
                    best_sol = c2.deepcopy()
                eval_count += 1
                str_eval += "{}: {} \n".format(eval_count, best_sol.cost)

                all_encodings.append([Bitmap.decode_encoding(c1.gene), Numeric.check_validity(c1.cost)])
                all_encodings.append([Bitmap.decode_encoding(c2.gene), Numeric.check_validity(c2.cost)])

                # b. Perform Mutation
                c1 = GA_Bitmap.mutate(c1, mu, sigma)
                c2 = GA_Bitmap.mutate(c2, mu, sigma)

                # Evaluate First Offspring
                c1.cost = Bitmap.cost_func(c1.gene, attr_keys_spl, d_set)
                if c1.cost == 1:
                    invalid_count += 1
                if c1.cost < best_sol.cost:
                    best_sol = c1.deepcopy()
                eval_count += 1
                str_eval += "{}: {} \n".format(eval_count, best_sol.cost)

                # Evaluate Second Offspring
                c2.cost = Bitmap.cost_func(c2.gene, attr_keys_spl, d_set)
                if c2.cost == 1:
                    invalid_count += 1
                if c2.cost < best_sol.cost:
                    best_sol = c2.deepcopy()
                eval_count += 1
                str_eval += "{}: {} \n".format(eval_count, best_sol.cost)

                # c. Add Offsprings to c_pop
                c_pop.append(c1)
                c_pop.append(c2)

                all_encodings.append([Bitmap.decode_encoding(c1.gene), Numeric.check_validity(c1.cost)])
                all_encodings.append([Bitmap.decode_encoding(c2.gene), Numeric.check_validity(c2.cost)])

            # Merge, Sort and Select
            pop += c_pop
            pop = sorted(pop, key=lambda x: x.cost)
            pop = pop[0:n_pop]

            best_gp = validate_gp(d_set, Bitmap.decode_gp(attr_keys_spl, best_sol.gene))
            is_present = is_duplicate(best_gp, best_patterns)
            is_sub = check_anti_monotony(best_patterns, best_gp, subset=True)
            if is_present or is_sub:
                repeated += 1
            else:
                if best_gp.support >= min_supp:
                    best_patterns.append(best_gp)
                # else:
                #    best_sol.cost = 1

            try:
                # Show Iteration Information
                # Store Best Cost
                best_costs[it_count] = best_sol.cost
                best_genes.append(best_sol.gene)
                # print("Iteration {}: Best Cost = {}".format(it_count, best_costs[it_count]))
                str_iter += "{}: {} \n".format(it_count, best_costs[it_count])
            except IndexError:
                pass
            it_count += 1

        # Output
        out = structure()
        out.pop = pop
        out.best_sol = best_sol
        out.best_costs = best_costs
        out.best_patterns = best_patterns
        out.invalid_pattern_count = invalid_count
        out.total_candidates = all_encodings
        out.str_iterations = str_iter
        out.str_evaluations = str_eval
        out.iteration_count = it_count
        out.max_iteration = max_iteration
        out.cost_evaluations = eval_count
        out.n_pop = n_pop
        out.pc = pc
        out.titles = d_set.titles
        out.col_count = d_set.col_count
        out.row_count = d_set.row_count
        return out

    @staticmethod
    def crossover(p1, p2, gamma=0.1):
        c1 = p1.deepcopy()
        c2 = p2.deepcopy()
        alpha = np.random.uniform(-gamma, 1 + gamma, c1.gene.shape[1])
        c1.gene = alpha * p1.gene + (1 - alpha) * p2.gene
        c2.gene = alpha * p2.gene + (1 - alpha) * p1.gene
        return c1, c2

    @staticmethod
    def mutate(x, mu, sigma):
        y = x.deepcopy()
        flag = np.random.rand(*x.gene.shape) <= mu
        ind = flag  # np.argwhere(flag)
        y.gene += sigma * np.random.rand(*ind.shape)
        return y

    @staticmethod
    def execute(f_path, min_supp, cores, max_iteration, n_pop, pc, gamma, mu, sigma, visuals):
        try:
            if cores > 1:
                num_cores = cores
            else:
                num_cores = sgp.get_num_cores()

            out = GA_Bitmap.run(f_path, min_supp, max_iteration, n_pop, pc, gamma, mu, sigma)
            list_gp = out.best_patterns

            wr_line = "Algorithm: GA-GRAANK (v1.0)\n"
            wr_line += "Search Space: Bitmap\n"
            wr_line += "No. of (dataset) attributes: " + str(out.col_count) + '\n'
            wr_line += "No. of (dataset) tuples: " + str(out.row_count) + '\n'
            wr_line += "Population size: " + str(out.n_pop) + '\n'
            wr_line += "PC: " + str(out.pc) + '\n'
            wr_line += "Gamma: " + str(gamma) + '\n'
            wr_line += "Mu: " + str(mu) + '\n'
            wr_line += "Sigma: " + str(sigma) + '\n'

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
    pbounds = {'data_src': (0, 0), 'min_supp': (0.5, 0.5), 'max_iteration': (1, 10), 'n_pop': (1, 20), 'pc': (0.1, 1),
               'gamma': (0.1, 0.9), 'mu': (0.1, 0.9), 'sigma': (0.1, 0.9)}

    optimizer = BayesianOptimization(
        f=GA_Numeric.run,
        pbounds=pbounds,
        random_state=1,
    )

    optimizer.maximize(
        init_points=10,
        n_iter=0,
    )
    return optimizer.max
