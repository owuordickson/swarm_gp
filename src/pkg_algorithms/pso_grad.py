# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@credits: "Thomas Runkler, and Anne Laurent,"
@license: "MIT"
@version: "2.0"
@email: "owuordickson@gmail.com"
@created: "29 April 2021"
@modified: "23 July 2021"

Breath-First Search for gradual patterns using Particle Swarm Optimization (PSO-GRAANK).
PSO is used to learn gradual pattern candidates.

CHANGES:
1. uses normal functions
2. updated fitness function to use Binary Array of GPs
3. used decimal <-> binary conversion for best_position <-> best_gradual_pattern (item-set combination)


"""
import numpy as np
import random
from ypstruct import structure

from .shared.gp import GI, GP
from .shared.dataset_bfs import Dataset
from .shared.profile import Profile


max_evals = 0
eval_count = 0
str_eval = ''


def run_particle_swarm(f_path, min_supp, max_iteration, max_evaluations, n_particles, velocity, coef_p, coef_g, nvar):
    global max_evals
    max_evals = max_evaluations

    # Prepare data set
    d_set = Dataset(f_path, min_supp)
    d_set.init_gp_attributes()
    # self.target = 1
    # self.target_error = 1e-6
    attr_keys = [GI(x[0], x[1].decode()).as_string() for x in d_set.valid_bins[:, 0]]

    if d_set.no_bins:
        return []

    it_count = 0
    var_min = 0
    var_max = int(''.join(['1']*len(attr_keys)), 2)

    # Empty particle template
    empty_particle = structure()
    empty_particle.position = None
    empty_particle.fitness = None

    # Initialize Population
    particle_pop = empty_particle.repeat(n_particles)
    for i in range(n_particles):
        particle_pop[i].position = random.randrange(var_min, var_max)
        particle_pop[i].fitness = 1

    pbest_pop = particle_pop.copy()
    gbest_particle = pbest_pop[0]

    # Best particle (ever found)
    best_particle = empty_particle.deepcopy()
    best_particle.position = gbest_particle.position
    best_particle.fitness = cost_func(best_particle.position, attr_keys, d_set)

    velocity_vector = np.zeros(n_particles)
    best_fitness_arr = np.empty(max_iteration)
    best_patterns = []
    str_iter = ''

    repeated = 0
    while eval_count < max_evaluations:
        # while it_count < max_iteration:
        # while repeated < 1:
        for i in range(n_particles):
            # UPDATED
            if particle_pop[i].position < var_min or particle_pop[i].position > var_max:
                particle_pop[i].fitness = 1
            else:
                particle_pop[i].fitness = cost_func(particle_pop[i].position, attr_keys, d_set)

            if pbest_pop[i].fitness > particle_pop[i].fitness:
                pbest_pop[i].fitness = particle_pop[i].fitness
                pbest_pop[i].position = particle_pop[i].position

            if gbest_particle.fitness > particle_pop[i].fitness:
                gbest_particle.fitness = particle_pop[i].fitness
                gbest_particle.position = particle_pop[i].position
        # if abs(gbest_fitness_value - self.target) < self.target_error:
        #    break
        if best_particle.fitness > gbest_particle.fitness:
            best_particle = gbest_particle.deepcopy()

        for i in range(n_particles):
            new_velocity = (velocity * velocity_vector[i]) + \
                           (coef_p * random.random()) * (pbest_pop[i].position - particle_pop[i].position) + \
                           (coef_g * random.random()) * (gbest_particle.position - particle_pop[i].position)
            particle_pop[i].position = particle_pop[i].position + new_velocity

        best_gp = validate_gp(d_set, decode_gp(attr_keys, best_particle.position))
        is_present = is_duplicate(best_gp, best_patterns)
        is_sub = check_anti_monotony(best_patterns, best_gp, subset=True)
        if is_present or is_sub:
            repeated += 1
        else:
            if best_gp.support >= min_supp:
                best_patterns.append(best_gp)
            # else:
            #    best_particle.fitness = 1

        try:
            # Show Iteration Information
            best_fitness_arr[it_count] = best_particle.fitness
            str_iter += "{}: {} \n".format(it_count, best_particle.fitness)
        except IndexError:
            pass
        it_count += 1

    # Output
    out = structure()
    out.pop = particle_pop
    out.best_costs = best_fitness_arr
    out.gbest_position = gbest_particle.position
    out.best_patterns = best_patterns
    out.str_iterations = str_iter
    out.iteration_count = it_count
    out.max_iteration = max_iteration
    out.str_evaluations = str_eval
    out.cost_evaluations = eval_count
    out.n_particles = n_particles
    out.W = velocity
    out.c1 = coef_p
    out.c2 = coef_g

    out.titles = d_set.titles
    out.col_count = d_set.col_count
    out.row_count = d_set.row_count

    return out


def cost_func(position, attr_keys, d_set):
    pattern = decode_gp(attr_keys, position)
    temp_bin = np.array([])
    for gi in pattern.gradual_items:
        arg = np.argwhere(np.isin(d_set.valid_bins[:, 0], gi.gradual_item))
        if len(arg) > 0:
            i = arg[0][0]
            valid_bin = d_set.valid_bins[i]
            if temp_bin.size <= 0:
                temp_bin = valid_bin[1].copy()
            else:
                temp_bin = np.multiply(temp_bin, valid_bin[1])
    bin_sum = np.sum(temp_bin)
    if bin_sum > 0:
        cost = (1 / bin_sum)
    else:
        cost = 1

    global str_eval
    global eval_count
    if eval_count < max_evals:
        eval_count += 1
        str_eval += "{}: {} \n".format(eval_count, cost)

    return cost


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


def execute(f_path, min_supp, cores, max_iteration, max_evaluations, n_particles, velocity, coef_p, coef_g, nvar):
    try:
        if cores > 1:
            num_cores = cores
        else:
            num_cores = Profile.get_num_cores()

        out = run_particle_swarm(f_path, min_supp, max_iteration, max_evaluations, n_particles, velocity, coef_p,
                                 coef_g, nvar)
        list_gp = out.best_patterns

        # Results
        Profile.plot_curve(out, 'Pattern Swarm Algorithm (PSO)')

        wr_line = "Algorithm: PSO-GRAANK (v2.0)\n"
        wr_line += "No. of (dataset) attributes: " + str(out.col_count) + '\n'
        wr_line += "No. of (dataset) tuples: " + str(out.row_count) + '\n'
        wr_line += "Velocity coeff.: " + str(out.W) + '\n'
        wr_line += "C1 coeff.: " + str(out.c1) + '\n'
        wr_line += "C2 coeff.: " + str(out.c2) + '\n'
        wr_line += "No. of particles: " + str(out.n_particles) + '\n'
        wr_line += "Minimum support: " + str(min_supp) + '\n'
        wr_line += "Number of cores: " + str(num_cores) + '\n'
        wr_line += "Number of patterns: " + str(len(list_gp)) + '\n'
        wr_line += "Number of iterations: " + str(out.iteration_count) + '\n'
        wr_line += "Number of cost evaluations: " + str(out.cost_evaluations) + '\n\n'

        for txt in out.titles:
            try:
                wr_line += (str(txt.key) + '. ' + str(txt.value.decode()) + '\n')
            except AttributeError:
                wr_line += (str(txt[0]) + '. ' + str(txt[1].decode()) + '\n')

        wr_line += str("\nFile: " + f_path + '\n')
        wr_line += str("\nPattern : Support" + '\n')

        for gp in list_gp:
            wr_line += (str(gp.to_string()) + ' : ' + str(round(gp.support, 3)) + '\n')

        # wr_line += '\n\n' + "Iteration: Best Cost" + '\n'
        # wr_line += out.str_iterations
        wr_line += '\n\n' + "Evaluation: Cost" + '\n'
        wr_line += out.str_evaluations
        return wr_line
    except ArithmeticError as error:
        wr_line = "Failed: " + str(error)
        print(error)
        return wr_line
