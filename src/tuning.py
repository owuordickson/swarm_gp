from bayes_opt import BayesianOptimization
from pkg_algorithms import aco_grad, ga_grad, pso_grad, prs_grad, pls_grad


def tune_ga():
    pbounds = {'data_src': (0, 0), 'min_supp': (0.5, 0.5), 'max_iteration': (1, 10), 'n_pop': (1, 20), 'pc': (0.1, 1),
               'gamma': (0.1, 0.9), 'mu': (0.1, 0.9), 'sigma': (0.1, 0.9)}

    optimizer = BayesianOptimization(
        f=ga_grad.run_genetic_algorithm,
        pbounds=pbounds,
        random_state=1,
    )

    optimizer.maximize(
        init_points=10,
        n_iter=0,
    )
    print(optimizer.max)


def tune_pso():
    pbounds = {'data_src': (0, 0), 'min_supp': (0.5, 0.5), 'max_iteration': (1, 10), 'n_particles': (1, 20),
               'velocity': (0.1, 1), 'coef_p': (0.1, 0.9), 'coef_g': (0.1, 0.9)}

    optimizer = BayesianOptimization(
        f=pso_grad.run_particle_swarm,
        pbounds=pbounds,
        random_state=1,
    )

    optimizer.maximize(
        init_points=10,
        n_iter=0,
    )
    print(optimizer.max)


def tune_pls():
    pbounds = {'data_src': (0, 0), 'min_supp': (0.5, 0.5), 'max_iteration': (1, 10), 'step_size': (0.1, 1),
               'nvar': (1, 1)}

    optimizer = BayesianOptimization(
        f=pls_grad.run_hill_climbing,
        pbounds=pbounds,
        random_state=1,
    )

    optimizer.maximize(
        init_points=10,
        n_iter=0,
    )
    print(optimizer.max)


def tune_prs():
    pbounds = {'data_src': (0, 0), 'min_supp': (0.5, 0.5), 'max_iteration': (1, 10), 'nvar': (1, 1)}

    optimizer = BayesianOptimization(
        f=prs_grad.run_pure_random_search,
        pbounds=pbounds,
        random_state=1,
    )

    optimizer.maximize(
        init_points=10,
        n_iter=0,
    )
    print(optimizer.max)


tune_ga()
