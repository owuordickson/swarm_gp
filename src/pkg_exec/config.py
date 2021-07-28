# -*- coding: utf-8 -*-

# Configurations for Gradual Patterns:
MIN_SUPPORT = 0.5
CPU_CORES = 4
# DATASET = "../../data/DATASET.csv"
DATASET = "../../data/hcv_data.csv"
ALGORITHM = 'prs'

# Uncomment for Main:
DATASET = "../data/hcv_data.csv"

# Uncomment for Terminal:
# DATASET = "data/DATASET.csv"

# Global Configurations
MAX_ITERATIONS = 100
N_VAR = 1  # DO NOT CHANGE

# ACO-GRAD Configurations:
EVAPORATION_FACTOR = 0.1

# GA-GRAD Configurations:
N_POPULATION = 5
PC = 0.9
GAMMA = 0.1  # Cross-over
MU = 0.1  # Mutation
SIGMA = 0.5  # Mutation

# PSO-GRAD Configurations:
VELOCITY = 0.5
PERSONAL_COEFF = 0.1
GLOBAL_COEFF = 0.9
TARGET = 1
TARGET_ERROR = 1e-6
N_PARTICLES = 5

# PLS-GRAD Configurations
STEP_SIZE = 0.1