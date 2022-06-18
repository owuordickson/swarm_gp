# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@credits: "Thomas Runkler, and Anne Laurent,"
@license: "MIT"
@version: "2.0"
@email: "owuordickson@gmail.com"
@created: "29 April 2021"
@modified: "07 September 2021"

Breath-First Search for gradual patterns using Particle Swarm Optimization (PSO-GRAANK).
PSO is used to learn gradual pattern candidates.

CHANGES:
1. uses normal functions
2. updated fitness function to use Binary Array of GPs
3. uses rank order search space


"""
import numpy as np
import random
from bayes_opt import BayesianOptimization
from ypstruct import structure
import so4gp as sgp

from .shared.gp import GI, validate_gp, is_duplicate, check_anti_monotony
from .shared.dataset import Dataset
from .shared.search_spaces import Bitmap, Numeric


class PSO_Numeric:

    @staticmethod
    def run(data_src, min_supp, max_iteration, n_particles, velocity, coef_p, coef_g):
        max_iteration = int(max_iteration)
        n_particles = int(n_particles)

        # Prepare data set
        d_set = Dataset(data_src, min_supp)
        d_set.init_gp_attributes()
        # self.target = 1
        # self.target_error = 1e-6
        attr_keys = [GI(x[0], x[1].decode()).as_string() for x in d_set.valid_bins[:, 0]]

        if d_set.no_bins:
            return []

        it_count = 0
        eval_count = 0
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
        best_particle.fitness = Numeric.cost_func(best_particle.position, attr_keys, d_set)

        velocity_vector = np.ones(n_particles)
        best_fitness_arr = np.empty(max_iteration)
        best_patterns = []
        str_iter = ''
        str_eval = ''

        invalid_count = 0
        all_encodings = []

        repeated = 0
        while it_count < max_iteration:
            # while eval_count < max_evaluations:
            # while repeated < 1:
            for i in range(n_particles):
                # UPDATED
                if particle_pop[i].position < var_min or particle_pop[i].position > var_max:
                    particle_pop[i].fitness = 1
                else:
                    particle_pop[i].fitness = Numeric.cost_func(particle_pop[i].position, attr_keys, d_set)
                    if particle_pop[i].fitness == 1:
                        invalid_count += 1
                    eval_count += 1
                    str_eval += "{}: {} \n".format(eval_count, particle_pop[i].fitness)
                    all_encodings.append([particle_pop[i].position, Numeric.check_validity(particle_pop[i].fitness)])

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

            best_gp = validate_gp(d_set, Numeric.decode_gp(attr_keys, best_particle.position))
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

        # Parameter Tuning - Output
        if data_src == 0.0:
            return 1/best_particle.fitness

        # Output
        out = structure()
        out.pop = particle_pop
        out.best_costs = best_fitness_arr
        out.gbest_position = gbest_particle.position
        out.best_patterns = best_patterns
        out.invalid_pattern_count = invalid_count
        out.total_candidates = all_encodings
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

    @staticmethod
    def execute(f_path, min_supp, cores, max_iteration, n_particles, velocity, coef_p, coef_g, visuals):
        try:
            if cores > 1:
                num_cores = cores
            else:
                num_cores = sgp.get_num_cores()

            out = PSO_Numeric.run(f_path, min_supp, max_iteration, n_particles, velocity, coef_p, coef_g)
            list_gp = out.best_patterns

            wr_line = "Algorithm: PSO-GRAANK (v2.0)\n"
            wr_line += "Search Space: Numeric\n"
            wr_line += "No. of (dataset) attributes: " + str(out.col_count) + '\n'
            wr_line += "No. of (dataset) tuples: " + str(out.row_count) + '\n'
            wr_line += "Particle population: " + str(out.n_particles) + '\n'
            wr_line += "Velocity coeff.: " + str(out.W) + '\n'
            wr_line += "Personal coeff.: " + str(out.c1) + '\n'
            wr_line += "Global coeff.: " + str(out.c2) + '\n'

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


class PSO_Bitmap:

    def __int__(self):
        pass

    @staticmethod
    def run(f_path, min_supp, max_iteration, n_particles, velocity, coef_p, coef_g):
        # Prepare data set
        d_set = Dataset(f_path, min_supp)
        d_set.init_gp_attributes()
        attr_keys = [GI(x[0], x[1].decode()).as_string() for x in d_set.valid_bins[:, 0]]
        attr_keys_spl = [attr_keys[x:x + 2] for x in range(0, len(attr_keys), 2)]

        if d_set.no_bins:
            return []

        it_count = 0
        eval_count = 0

        # Empty particle template
        empty_particle = structure()
        empty_particle.position = None
        empty_particle.fitness = None

        # Initialize Population
        particle_pop = empty_particle.repeat(n_particles)
        for i in range(n_particles):
            particle_pop[i].position = Bitmap.build_gp_gene(attr_keys_spl)
            particle_pop[i].fitness = 1

        pbest_pop = particle_pop.copy()
        gbest_particle = pbest_pop[0]

        # Best particle (ever found)
        best_particle = empty_particle.deepcopy()
        best_particle.position = gbest_particle.position
        best_particle.fitness = Bitmap.cost_func(best_particle.position, attr_keys_spl, d_set)

        velocity_vector = np.ones(n_particles)
        best_fitness_arr = np.empty(max_iteration)
        best_patterns = []
        str_iter = ''
        str_eval = ''

        invalid_count = 0
        all_encodings = []

        repeated = 0
        while it_count < max_iteration:
            # while eval_count < max_evaluations:
            # while repeated < 1:
            for i in range(n_particles):
                particle_pop[i].fitness = Bitmap.cost_func(particle_pop[i].position, attr_keys_spl, d_set)
                if particle_pop[i].fitness == 1:
                    invalid_count += 1
                eval_count += 1
                str_eval += "{}: {} \n".format(eval_count, particle_pop[i].fitness)
                all_encodings.append([Bitmap.decode_encoding(particle_pop[i].position),
                                      Numeric.check_validity(particle_pop[i].fitness)])

                if pbest_pop[i].fitness > particle_pop[i].fitness:
                    pbest_pop[i].fitness = particle_pop[i].fitness
                    pbest_pop[i].position = particle_pop[i].position

                if gbest_particle.fitness > particle_pop[i].fitness:
                    gbest_particle.fitness = particle_pop[i].fitness
                    gbest_particle.position = particle_pop[i].position

            if best_particle.fitness > gbest_particle.fitness:
                best_particle = gbest_particle.deepcopy()

            for i in range(n_particles):
                new_velocity = (velocity * velocity_vector[i]) + \
                               (coef_p * random.random()) * (pbest_pop[i].position - particle_pop[i].position) + \
                               (coef_g * random.random()) * (gbest_particle.position - particle_pop[i].position)
                particle_pop[i].position = particle_pop[i].position + new_velocity

            best_gp = validate_gp(d_set, Bitmap.decode_gp(attr_keys_spl, best_particle.position))
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
        out.invalid_pattern_count = invalid_count
        out.total_candidates = all_encodings
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

    @staticmethod
    def execute(f_path, min_supp, cores, max_iteration, n_particles, velocity, coef_p, coef_g, visuals):
        try:
            if cores > 1:
                num_cores = cores
            else:
                num_cores = sgp.get_num_cores()

            out = PSO_Bitmap.run(f_path, min_supp, max_iteration, n_particles, velocity, coef_p, coef_g)
            list_gp = out.best_patterns

            wr_line = "Algorithm: PSO-GRAANK (v1.0)\n"
            wr_line += "Search Space: Bitmap\n"
            wr_line += "No. of (dataset) attributes: " + str(out.col_count) + '\n'
            wr_line += "No. of (dataset) tuples: " + str(out.row_count) + '\n'
            wr_line += "Particle population: " + str(out.n_particles) + '\n'
            wr_line += "Velocity coeff.: " + str(out.W) + '\n'
            wr_line += "Personal coeff.: " + str(out.c1) + '\n'
            wr_line += "Global coeff.: " + str(out.c2) + '\n'

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
    pbounds = {'data_src': (0, 0), 'min_supp': (0.5, 0.5), 'max_iteration': (1, 10), 'n_particles': (1, 20),
               'velocity': (0.1, 1), 'coef_p': (0.1, 0.9), 'coef_g': (0.1, 0.9)}

    optimizer = BayesianOptimization(
        f=PSO_Numeric.run,
        pbounds=pbounds,
        random_state=1,
    )

    optimizer.maximize(
        init_points=10,
        n_iter=0,
    )
    return optimizer.max
