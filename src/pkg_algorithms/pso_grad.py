# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@credits: "Thomas Runkler, Edmond Menya, and Anne Laurent,"
@license: "MIT"
@version: "1.0"
@email: "owuordickson@gmail.com"
@created: "29 April 2021"
@modified: "23 July 2021"

Breath-First Search for gradual patterns (PSO-GRAANK)

CHANGES:
1. uses normal functions
2. updated fitness function to use Binary Array of GPs


"""
import numpy as np
import random
from ypstruct import structure

from .shared.gp import GI, GP
from .shared.dataset_bfs import Dataset
from .shared.profile import Profile
from .shared import config as cfg


def run_particle_swarm(f_path, min_supp, max_iteration=cfg.MAX_ITERATIONS, n_particles=cfg.N_PARTICLES,
                       velocity=cfg.VELOCITY, coeff_p=cfg.PERSONAL_COEFF, coeff_g=cfg.GLOBAL_COEFF):
    # Prepare data set
    d_set = Dataset(f_path, min_supp)
    d_set.init_gp_attributes()
    # self.target = 1
    # self.target_error = 1e-6
    attr_keys = [GI(x[0], x[1].decode()).as_string() for x in d_set.valid_bins[:, 0]]

    if d_set.no_bins:
        return []

    it_count = 0

    # Empty particle template
    empty_particle = structure()
    empty_particle.position = None
    empty_particle.fitness = None

    # Best particle (ever found)
    best_particle = empty_particle.deepcopy()
    best_particle.fitness = float('inf')

    # Initialize Population
    particle_pop = empty_particle.repeat(n_particles)
    for i in range(n_particles):
        particle_pop[i].position = build_gp_position(attr_keys)
        particle_pop[i].fitness = float('inf')  # fitness_function(pbest_pop[i].position, attr_keys, d_set)
        # if pbest_pop[i].fitness < best_particle.fitness:
        #     best_particle = pbest_pop[i].deepcopy()

    pbest_pop = particle_pop.copy()
    gbest_particle = pbest_pop[0]

    velocity_vector = ([np.zeros((len(attr_keys),)) for _ in range(n_particles)])
    best_fitness_arr = np.empty(max_iteration)
    best_patterns = []
    str_plt = ''

    repeated = 0
    while it_count < max_iteration:
        # while repeated < 1:
        for i in range(n_particles):
            # UPDATED
            particle_pop[i].fitness = fitness_function(particle_pop[i].position, attr_keys, d_set)
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
            x1 = np.dot(velocity, velocity_vector[i])
            x2 = np.dot(coeff_p, random.random())
            x3 = np.dot(coeff_g, random.random())

            x4 = (pbest_pop[i].position - particle_pop[i].position)
            x5 = (gbest_particle.position - particle_pop[i].position)
            new_velocity = x1 + np.dot(x2, x4) + np.dot(x3, x5)
            particle_pop[i].position = particle_pop[i].position + new_velocity

        best_gp = validate_gp(d_set, decode_gp(attr_keys, best_particle.position))
        is_present = is_duplicate(best_gp, best_patterns)
        is_sub = check_anti_monotony(best_patterns, best_gp, subset=True)
        if is_present or is_sub:
            repeated += 1
        else:
            if best_gp.support >= min_supp:
                best_patterns.append(best_gp)
            else:
                best_particle.fitness = 1

        try:
            # Show Iteration Information
            best_fitness_arr[it_count] = best_particle.fitness  # best_fitness
            # print("Iteration {}: Best Position = {}".format(it_count, best_fitness_arr[it_count]))
            str_plt += "Iteration {}: Best Fitness Value: {} \n".format(it_count, best_fitness_arr[it_count])
        except IndexError:
            pass
        it_count += 1

    # Output
    out = structure()
    out.pop = particle_pop
    out.best_pos = best_fitness_arr
    out.gbest_position = gbest_particle.position
    out.best_patterns = best_patterns

    out.best_patterns = best_patterns
    out.str_iterations = str_plt
    out.iteration_count = it_count
    out.max_iteration = max_iteration
    out.n_particles = n_particles
    out.W = velocity
    out.c1 = coeff_p
    out.c2 = coeff_g

    out.titles = d_set.titles
    out.col_count = d_set.col_count
    out.row_count = d_set.row_count

    return out


def build_gp_position(attr_keys):
    attr = attr_keys
    temp_gene = np.random.choice(a=[0, 1], size=(len(attr),))
    return temp_gene


def fitness_function(position, attr_keys, d_set):
    # if gp is None:
    #    return np.inf
    # else:
    #    if gp.support <= thd_supp:
    #        return np.inf
    #    return round((1 / gp.support), 2)
    print(position)
    bin_sum = compute_bin_sum(d_set, decode_gp(attr_keys, position))
    if bin_sum > 0:
        cost = (1 / bin_sum)
    else:
        cost = 1
    return cost


def decode_gp(attr_keys, position):
    temp_gp = GP()
    if position is None:
        return temp_gp
    for i in range(position.size):
        gene_val = round(position[i])
        if gene_val >= 1:
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
    for gi in pattern.gradual_items:
        arg = np.argwhere(np.isin(d_set.valid_bins[:, 0], gi.gradual_item))
        if len(arg) > 0:
            i = arg[0][0]
            valid_bin = d_set.valid_bins[i]
            if temp_bin.size <= 0:
                temp_bin = valid_bin[1].copy()
            else:
                temp_bin = np.multiply(temp_bin, valid_bin[1])
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

        out = run_particle_swarm(f_path, min_supp)
        list_gp = out.best_patterns

        # Results
        # plt.plot(out.best_pos)
        # plt.semilogy(out.best_pos)
        # plt.xlim(0, pso.max_it)
        # plt.xlabel('Iterations')
        # plt.ylabel('Global Best Position')
        # plt.title('Pattern Swarm Algorithm (PSO)')
        # plt.grid(True)
        # plt.show()

        wr_line = "Algorithm: PSO-GRAANK (v1.0)\n"
        wr_line += "No. of (dataset) attributes: " + str(out.col_count) + '\n'
        wr_line += "No. of (dataset) tuples: " + str(out.row_count) + '\n'
        wr_line += "Velocity coeff.: " + str(out.W) + '\n'
        wr_line += "C1 coeff.: " + str(out.c1) + '\n'
        wr_line += "C2 coeff.: " + str(out.c2) + '\n'
        wr_line += "No. of particles: " + str(out.n_particles) + '\n'
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
