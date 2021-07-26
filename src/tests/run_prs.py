import random
import numpy as np


def optimize(function, dimensions, lower_boundary, upper_boundary, max_iter, maximize=False):
    best_position = np.array([float()] * dimensions)

    best_costs = np.empty(max_iter)
    it_count = 0

    for i in range(dimensions):
        best_position[i] = random.uniform(lower_boundary[i], upper_boundary[i])

    # for it_count in range(max_iter):
    while it_count < max_iter:

        cost_1 = function(best_position)
        best_cost = cost_1

        new_position = [lower_boundary[d] + random.random() * (upper_boundary[d] - lower_boundary[d]) for d in range(dimensions)]

        if np.greater_equal(new_position, lower_boundary).all() and np.less_equal(new_position, upper_boundary).all():
            cost_2 = function(new_position)
        elif maximize:
            cost_2 = -100000.0
        else:
            cost_2 = 100000.0

        if cost_2 > cost_1 and maximize:
            best_position = new_position
            best_cost = cost_2
        elif cost_2 < cost_1 and not maximize:
            best_position = new_position
            best_cost = cost_2

        try:
            # Show Iteration Information
            # Store Best Cost
            best_costs[it_count] = best_cost
            # str_plt += "Iteration {}: Best Cost: {} \n".format(it_count, best_costs[it_count])
            print("Iteration {}: Best Cost: {} \n".format(it_count, best_costs[it_count]))
        except IndexError:
            pass
        it_count += 1

    best_fitness = function(best_position)

    return best_fitness, best_position


def your_function(x):
    return -(x[0] ** 2 + x[1] ** 2) + 4


a, b = optimize(function=your_function, dimensions=2, lower_boundary=[-14, -14], upper_boundary=[10, 10],
                max_iter=100, maximize=True)
print(a, ",", b)

