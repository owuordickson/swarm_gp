# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@created: "03 May 2021"
Usage:
    $python init_acograd.py -f ../data/DATASET.csv -s 0.5
Description:
    f -> file path (CSV)
    s -> minimum support
"""

import sys
from optparse import OptionParser
import config as cfg
from pkg_algorithms import ga_grad, pso_grad, prs_grad, pls_grad, aco_grad, graank, lcm_gp


def get_quick_mem_use(s_shot, key_type='lineno'):
    import tracemalloc
    s_shot = s_shot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = s_shot.statistics(key_type)
    total = sum(stat.size for stat in top_stats)
    wr_line = ("Total allocated memory size: %.1f KiB" % (total / 1024))
    return wr_line


def configure_parameters(fpath):
    if "DATASET" in fpath:
        print("test set")
    elif "hungary_chickenpox" in fpath:
        # GA-GRAD Configurations:
        cfg.N_POPULATION = 12
        cfg.PC = 0.913
        cfg.GAMMA = 529  # Cross-over
        cfg.MU = 0.856  # Mutation
        cfg.SIGMA = 0.209  # Mutation

        # PSO-GRAD Configurations:
        cfg.N_PARTICLES = 19
        cfg.VELOCITY = 0.628
        cfg.PERSONAL_COEFF = 0.14
        cfg.GLOBAL_COEFF = 0.431

        # PLS-GRAD Configurations
        cfg.STEP_SIZE = 0.232
    elif "air_quality" in fpath:
        # GA-GRAD Configurations:
        cfg.N_POPULATION = 20
        cfg.PC = 0.382
        cfg.GAMMA = 547  # Cross-over
        cfg.MU = 0.741  # Mutation
        cfg.SIGMA = 0.654  # Mutation

        # PSO-GRAD Configurations:
        cfg.N_PARTICLES = 19
        cfg.VELOCITY = 0.628
        cfg.PERSONAL_COEFF = 0.14
        cfg.GLOBAL_COEFF = 0.431

        # PLS-GRAD Configurations
        cfg.STEP_SIZE = 0.232
    elif "aps_2k" in fpath:
        # GA-GRAD Configurations:
        cfg.N_POPULATION = 20
        cfg.PC = 0.382
        cfg.GAMMA = 547  # Cross-over
        cfg.MU = 0.741  # Mutation
        cfg.SIGMA = 0.654  # Mutation

        # PSO-GRAD Configurations:
        cfg.N_PARTICLES = 5
        cfg.VELOCITY = 0.9
        cfg.PERSONAL_COEFF = 0.01
        cfg.GLOBAL_COEFF = 0.9

        # PLS-GRAD Configurations
        cfg.STEP_SIZE = 0.125
    elif "breast_cancer" in fpath:
        # GA-GRAD Configurations:
        cfg.N_POPULATION = 8
        cfg.PC = 0.776
        cfg.GAMMA = 746  # Cross-over
        cfg.MU = 0.842  # Mutation
        cfg.SIGMA = 0.681  # Mutation

        # PSO-GRAD Configurations:
        cfg.N_PARTICLES = 19
        cfg.VELOCITY = 0.628
        cfg.PERSONAL_COEFF = 0.14
        cfg.GLOBAL_COEFF = 0.431

        # PLS-GRAD Configurations
        cfg.STEP_SIZE = 0.278
    elif "c2k" in fpath:
        # GA-GRAD Configurations:
        cfg.N_POPULATION = 20
        cfg.PC = 0.382
        cfg.GAMMA = 547  # Cross-over
        cfg.MU = 0.741  # Mutation
        cfg.SIGMA = 0.654  # Mutation

        # PSO-GRAD Configurations:
        cfg.N_PARTICLES = 19
        cfg.VELOCITY = 0.628
        cfg.PERSONAL_COEFF = 0.14
        cfg.GLOBAL_COEFF = 0.431

        # PLS-GRAD Configurations
        cfg.STEP_SIZE = 0.723
    elif "directio8k" in fpath:
        # GA-GRAD Configurations:
        cfg.N_POPULATION = 8
        cfg.PC = 0.776
        cfg.GAMMA = 746  # Cross-over
        cfg.MU = 0.842  # Mutation
        cfg.SIGMA = 0.681  # Mutation

        # PSO-GRAD Configurations:
        cfg.N_PARTICLES = 20
        cfg.VELOCITY = 0.77
        cfg.PERSONAL_COEFF = 0.649
        cfg.GLOBAL_COEFF = 0.352

        # PLS-GRAD Configurations
        cfg.STEP_SIZE = 0.217
    elif "hcv_data" in fpath:
        # GA-GRAD Configurations:
        cfg.N_POPULATION = 8
        cfg.PC = 0.776
        cfg.GAMMA = 746  # Cross-over
        cfg.MU = 0.842  # Mutation
        cfg.SIGMA = 0.681  # Mutation

        # PSO-GRAD Configurations:
        cfg.N_PARTICLES = 7
        cfg.VELOCITY = 0.359
        cfg.PERSONAL_COEFF = 0.731
        cfg.GLOBAL_COEFF = 0.324

        # PLS-GRAD Configurations
        cfg.STEP_SIZE = 0.89
    elif "Omnidir_site2k" in fpath:
        # GA-GRAD Configurations:
        cfg.N_POPULATION = 8
        cfg.PC = 0.776
        cfg.GAMMA = 746  # Cross-over
        cfg.MU = 0.842  # Mutation
        cfg.SIGMA = 0.681  # Mutation

        # PSO-GRAD Configurations:
        cfg.N_PARTICLES = 7
        cfg.VELOCITY = 0.359
        cfg.PERSONAL_COEFF = 0.731
        cfg.GLOBAL_COEFF = 0.324

        # PLS-GRAD Configurations
        cfg.STEP_SIZE = 0.278
    elif "power_consumption10k" in fpath:
        # GA-GRAD Configurations:
        cfg.N_POPULATION = 17
        cfg.PC = 0.116
        cfg.GAMMA = 527  # Cross-over
        cfg.MU = 0.649  # Mutation
        cfg.SIGMA = 0.7  # Mutation

        # PSO-GRAD Configurations:
        cfg.N_PARTICLES = 5
        cfg.VELOCITY = 0.890
        cfg.PERSONAL_COEFF = 0.417
        cfg.GLOBAL_COEFF = 0.376

        # PLS-GRAD Configurations
        cfg.STEP_SIZE = 0.232
    elif "" in fpath:
        pass


if __name__ == "__main__":
    if not sys.argv:
        algChoice = sys.argv[0]
        filePath = sys.argv[1]
        minSup = sys.argv[2]
        numCores = sys.argv[3]
        eVal = sys.argv[4]
        pcVal = sys.argv[5]
        vFactor = sys.argv[6]
        stepVal = sys.argv[7]
        pTune = sys.argv[8]
    else:
        optparser = OptionParser()
        optparser.add_option('-a', '--algorithmChoice',
                             dest='algChoice',
                             help='select algorithm',
                             default=cfg.ALGORITHM,
                             type='string')
        optparser.add_option('-f', '--inputFile',
                             dest='file',
                             help='path to file containing csv',
                             default=cfg.DATASET,
                             type='string')
        optparser.add_option('-s', '--minSupport',
                             dest='minSup',
                             help='minimum support value',
                             default=cfg.MIN_SUPPORT,
                             type='float')
        optparser.add_option('-c', '--cores',
                             dest='numCores',
                             help='number of cores',
                             default=cfg.CPU_CORES,
                             type='int')
        optparser.add_option('-e', '--eFactor',
                             dest='eVal',
                             help='evaporation factor (ACO)',
                             default=cfg.EVAPORATION_FACTOR,
                             type='float')
        optparser.add_option('-p', '--propOffsprings',
                             dest='pcVal',
                             help='proportion of children/offsprings (GA)',
                             default=cfg.PC,
                             type='float')
        optparser.add_option('-v', '--velocityFactor',
                             dest='vFactor',
                             help='velocity factor (PSO)',
                             default=cfg.VELOCITY,
                             type='float')
        optparser.add_option('-t', '--stepSize',
                             dest='stepVal',
                             help='step size (PLS)',
                             default=cfg.STEP_SIZE,
                             type='float')
        optparser.add_option('-x', '--parameterTuning',
                             dest='pTune',
                             help='Parameter Tuning',
                             default=cfg.TUNE_VAL,
                             type='int')
        (options, args) = optparser.parse_args()

        if options.file is None:
            print("Usage: $python3 main.py -a 'aco' -f filename.csv ")
            sys.exit('System will exit')
        else:
            filePath = options.file
        algChoice = options.algChoice
        minSup = options.minSup
        numCores = options.numCores
        eVal = options.eVal
        pcVal = options.pcVal
        vFactor = options.vFactor
        stepVal = options.stepVal
        pTune = options.pTune

    VISUAL = [0, 0, 0]
    if cfg.SHOW_P_MATRIX:
        VISUAL[0] = True
    if cfg.SHOW_EVALUATIONS:
        VISUAL[1] = True
    if cfg.SHOW_ITERATIONS:
        VISUAL[2] = True

    import time
    import tracemalloc
    import so4gp as sgp
    from pkg_algorithms.shared.dataset import CONF_SOURCE

    configure_parameters(filePath)

    if algChoice == 'ga':
        # GA-GRAANK
        if pTune == 1:
            res_txt = str(ga_grad.parameter_tuning()) + "\n"
            res_txt += str("\nFile: " + CONF_SOURCE)

            stamp = time.time()
            f_name = str('tune_ga' + str(stamp).replace('.', '', 1) + '.txt')
            sgp.write_file(res_txt, f_name, cfg.SAVE_RESULTS)
            print(res_txt)
        else:
            start = time.time()
            tracemalloc.start()
            if cfg.SEARCH_SPACE == 'bm':
                res_text = ga_grad.GA_Bitmap.execute(filePath, minSup, numCores, cfg.MAX_ITERATIONS, cfg.N_POPULATION,
                                                     pcVal, cfg.GAMMA, cfg.MU, cfg.SIGMA, VISUAL)
            else:
                res_text = ga_grad.GA_Numeric.execute(filePath, minSup, numCores, cfg.MAX_ITERATIONS, cfg.N_POPULATION,
                                                      pcVal, cfg.GAMMA, cfg.MU, cfg.SIGMA, VISUAL)
            snapshot = tracemalloc.take_snapshot()
            end = time.time()

            wr_text = ("Run-time: " + str(end - start) + " seconds\n")
            wr_text += (get_quick_mem_use(snapshot) + "\n")
            wr_text += str(res_text)
            f_name = str('res_ga' + str(end).replace('.', '', 1) + '.txt')
            sgp.write_file(wr_text, f_name, cfg.SAVE_RESULTS)
            print(wr_text)
    elif algChoice == 'pso':
        # PSO-GRAANK
        if pTune == 1:
            res_txt = str(pso_grad.parameter_tuning()) + "\n"
            res_txt += str("\nFile: " + CONF_SOURCE)

            stamp = time.time()
            f_name = str('tune_pso' + str(stamp).replace('.', '', 1) + '.txt')
            sgp.write_file(res_txt, f_name, cfg.SAVE_RESULTS)
            print(res_txt)
        else:
            start = time.time()
            tracemalloc.start()
            if cfg.SEARCH_SPACE == 'bm':
                res_text = pso_grad.PSO_Bitmap.execute(filePath, minSup, numCores, cfg.MAX_ITERATIONS, cfg.N_PARTICLES,
                                                       vFactor, cfg.PERSONAL_COEFF, cfg.GLOBAL_COEFF, VISUAL)
            else:
                res_text = pso_grad.PSO_Numeric.execute(filePath, minSup, numCores, cfg.MAX_ITERATIONS, cfg.N_PARTICLES,
                                                        vFactor, cfg.PERSONAL_COEFF, cfg.GLOBAL_COEFF, VISUAL)
            snapshot = tracemalloc.take_snapshot()
            end = time.time()

            wr_text = ("Run-time: " + str(end - start) + " seconds\n")
            wr_text += (get_quick_mem_use(snapshot) + "\n")
            wr_text += str(res_text)
            f_name = str('res_pso' + str(end).replace('.', '', 1) + '.txt')
            sgp.write_file(wr_text, f_name, cfg.SAVE_RESULTS)
            print(wr_text)
    elif algChoice == 'prs':
        # PRS-GRAANK
        if pTune == 1:
            res_txt = str(prs_grad.parameter_tuning()) + "\n"
            res_txt += str("\nFile: " + CONF_SOURCE)

            stamp = time.time()
            f_name = str('tune_prs' + str(stamp).replace('.', '', 1) + '.txt')
            sgp.write_file(res_txt, f_name, cfg.SAVE_RESULTS)
            print(res_txt)
        else:
            start = time.time()
            tracemalloc.start()
            if cfg.SEARCH_SPACE == 'bm':
                res_text = prs_grad.RS_Bitmap.execute(filePath, minSup, numCores, cfg.MAX_ITERATIONS, cfg.N_VAR, VISUAL)
            else:
                res_text = prs_grad.RS_Numeric.execute(filePath, minSup, numCores, cfg.MAX_ITERATIONS, cfg.N_VAR, VISUAL)
            snapshot = tracemalloc.take_snapshot()
            end = time.time()

            wr_text = ("Run-time: " + str(end - start) + " seconds\n")
            wr_text += (get_quick_mem_use(snapshot) + "\n")
            wr_text += str(res_text)
            f_name = str('res_prs' + str(end).replace('.', '', 1) + '.txt')
            sgp.write_file(wr_text, f_name, cfg.SAVE_RESULTS)
            print(wr_text)
    elif algChoice == 'pls':
        # PLS-GRAANK
        if pTune == 1:
            res_txt = str(pls_grad.parameter_tuning()) + "\n"
            res_txt += str("\nFile: " + CONF_SOURCE)

            stamp = time.time()
            f_name = str('tune_pls' + str(stamp).replace('.', '', 1) + '.txt')
            sgp.write_file(res_txt, f_name, cfg.SAVE_RESULTS)
            print(res_txt)
        else:
            start = time.time()
            tracemalloc.start()
            if cfg.SEARCH_SPACE == 'bm':
                res_text = pls_grad.LS_Bitmap.execute(filePath, minSup, numCores, cfg.MAX_ITERATIONS, stepVal,
                                                      cfg.N_VAR, VISUAL)
            else:
                res_text = pls_grad.LS_Numeric.execute(filePath, minSup, numCores, cfg.MAX_ITERATIONS, stepVal,
                                                       cfg.N_VAR, VISUAL)
            snapshot = tracemalloc.take_snapshot()
            end = time.time()

            wr_text = ("Run-time: " + str(end - start) + " seconds\n")
            wr_text += (get_quick_mem_use(snapshot) + "\n")
            wr_text += str(res_text)
            f_name = str('res_pls' + str(end).replace('.', '', 1) + '.txt')
            sgp.write_file(wr_text, f_name, cfg.SAVE_RESULTS)
            print(wr_text)
    elif algChoice == 'aco':
        # ACO-GRAANK
        start = time.time()
        tracemalloc.start()
        res_text = aco_grad.execute(filePath, minSup, numCores, cfg.EVAPORATION_FACTOR, cfg.MAX_ITERATIONS)
        snapshot = tracemalloc.take_snapshot()
        end = time.time()

        wr_text = ("Run-time: " + str(end - start) + " seconds\n")
        wr_text += (get_quick_mem_use(snapshot) + "\n")
        wr_text += str(res_text)
        f_name = str('res_aco' + str(end).replace('.', '', 1) + '.txt')
        sgp.write_file(wr_text, f_name, cfg.SAVE_RESULTS)
        print(wr_text)
    elif algChoice == 'gra':
        # GRAANK
        start = time.time()
        tracemalloc.start()
        res_text = graank.execute(filePath, minSup, numCores)
        snapshot = tracemalloc.take_snapshot()
        end = time.time()

        wr_text = ("Run-time: " + str(end - start) + " seconds\n")
        wr_text += (get_quick_mem_use(snapshot) + "\n")
        wr_text += str(res_text)
        f_name = str('res_graank' + str(end).replace('.', '', 1) + '.txt')
        sgp.write_file(wr_text, f_name, cfg.SAVE_RESULTS)
        print(wr_text)
    elif algChoice == 'lcm':
        # GRAANK
        start = time.time()
        tracemalloc.start()
        res_text = lcm_gp.execute(filePath, minSup, numCores)
        snapshot = tracemalloc.take_snapshot()
        end = time.time()

        wr_text = ("Run-time: " + str(end - start) + " seconds\n")
        wr_text += (get_quick_mem_use(snapshot) + "\n")
        wr_text += str(res_text)
        f_name = str('res_graank' + str(end).replace('.', '', 1) + '.txt')
        sgp.write_file(wr_text, f_name, cfg.SAVE_RESULTS)
        print(wr_text)
    else:
        print("Invalid Algorithm Choice!")
