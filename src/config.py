# -*- coding: utf-8 -*-

# Configurations for Gradual Patterns:
# INITIALIZATIONS = 3
ALGORITHM = 'aco'  # aco, ga, pso, prs, pls, gra, lcm
MIN_SUPPORT = 0.5
CPU_CORES = 4

# Uncomment for Main:
DATASET = "../data/DATASET.csv"
# DATASET = "../data/hcv_data.csv"
# DATASET = "../data/hungary_chickenpox.csv"
# DATASET = "../data/AirQualityUCI.csv"
# DATASET = "../data/c2k.csv"

# Uncomment for Terminal:
# DATASET = "data/hcv_data.csv"

# Global Swarm Configurations
MAX_ITERATIONS = 20
N_VAR = 1  # DO NOT CHANGE

# ACO-GRAD Configurations:
EVAPORATION_FACTOR = 0.5

# GA-GRAD Configurations:
N_POPULATION = 5
PC = 0.5
GAMMA = 1  # Cross-over
MU = 0.9  # Mutation
SIGMA = 0.9  # Mutation

# PSO-GRAD Configurations:
N_PARTICLES = 5
VELOCITY = 0.9  # higher values helps to move to next number in search space
PERSONAL_COEFF = 0.01
GLOBAL_COEFF = 0.9
TARGET = 1
TARGET_ERROR = 1e-6

# PLS-GRAD Configurations
STEP_SIZE = 0.5

# VISUALIZATIONS
SHOW_P_MATRIX = False  # ONLY FOR: aco
SHOW_EVALUATIONS = False  # FOR aco, prs, pls, pso
SHOW_ITERATIONS = True  # FOR aco, prs, pls, pso
SAVE_RESULTS = True  # FOR aco, prs, pls, pso

TUNE_VAL = 0

SEARCH_SPACE = 'bm'  # nu, bm
