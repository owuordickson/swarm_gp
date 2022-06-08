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
from pkg_algorithms import aco_grad, ga_grad, pso_grad, prs_grad, pls_grad

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
    from pkg_algorithms.shared.profile import Profile

    if algChoice == 'aco':
        # ACO-GRAANK
        start = time.time()
        tracemalloc.start()
        res_text = aco_grad.execute(filePath, minSup, numCores, eVal, cfg.MAX_ITERATIONS, VISUAL)
        snapshot = tracemalloc.take_snapshot()
        end = time.time()

        wr_text = ("Run-time: " + str(end - start) + " seconds\n")
        wr_text += (Profile.get_quick_mem_use(snapshot) + "\n")
        wr_text += str(res_text)
        f_name = str('res_aco' + str(end).replace('.', '', 1) + '.txt')
        Profile.write_file(wr_text, f_name, cfg.SAVE_RESULTS)
        print(wr_text)
    elif algChoice == 'ga':
        # GA-GRAANK
        if pTune == 1:
            res_json = ga_grad.parameter_tuning()
            stamp = time.time()
            f_name = str('tune_ga' + str(stamp).replace('.', '', 1) + '.txt')
            Profile.write_file(str(res_json), f_name, cfg.SAVE_RESULTS)
            print(res_json)
        else:
            start = time.time()
            tracemalloc.start()
            res_text = ga_grad.execute(filePath, minSup, numCores, cfg.MAX_ITERATIONS, cfg.MAX_EVALUATIONS,
                                       cfg.N_POPULATION, pcVal, cfg.GAMMA, cfg.MU, cfg.SIGMA, VISUAL)
            snapshot = tracemalloc.take_snapshot()
            end = time.time()

            wr_text = ("Run-time: " + str(end - start) + " seconds\n")
            wr_text += (Profile.get_quick_mem_use(snapshot) + "\n")
            wr_text += str(res_text)
            f_name = str('res_ga' + str(end).replace('.', '', 1) + '.txt')
            Profile.write_file(wr_text, f_name, cfg.SAVE_RESULTS)
            print(wr_text)
    elif algChoice == 'pso':
        # PSO-GRAANK
        if pTune == 1:
            res_json = pso_grad.parameter_tuning()
            stamp = time.time()
            f_name = str('tune_pso' + str(stamp).replace('.', '', 1) + '.txt')
            Profile.write_file(str(res_json), f_name, cfg.SAVE_RESULTS)
            print(res_json)
        else:
            start = time.time()
            tracemalloc.start()
            res_text = pso_grad.execute(filePath, minSup, numCores, cfg.MAX_ITERATIONS, cfg.MAX_EVALUATIONS,
                                        cfg.N_PARTICLES, vFactor, cfg.PERSONAL_COEFF, cfg.GLOBAL_COEFF, VISUAL)
            snapshot = tracemalloc.take_snapshot()
            end = time.time()

            wr_text = ("Run-time: " + str(end - start) + " seconds\n")
            wr_text += (Profile.get_quick_mem_use(snapshot) + "\n")
            wr_text += str(res_text)
            f_name = str('res_pso' + str(end).replace('.', '', 1) + '.txt')
            Profile.write_file(wr_text, f_name, cfg.SAVE_RESULTS)
            print(wr_text)
    elif algChoice == 'prs':
        # PRS-GRAANK
        if pTune == 1:
            res_json = prs_grad.parameter_tuning()
            stamp = time.time()
            f_name = str('tune_prs' + str(stamp).replace('.', '', 1) + '.txt')
            Profile.write_file(str(res_json), f_name, cfg.SAVE_RESULTS)
            print(res_json)
        else:
            start = time.time()
            tracemalloc.start()
            res_text = prs_grad.execute(filePath, minSup, numCores, cfg.MAX_ITERATIONS, cfg.MAX_EVALUATIONS, cfg.N_VAR,
                                        VISUAL)
            snapshot = tracemalloc.take_snapshot()
            end = time.time()

            wr_text = ("Run-time: " + str(end - start) + " seconds\n")
            wr_text += (Profile.get_quick_mem_use(snapshot) + "\n")
            wr_text += str(res_text)
            f_name = str('res_prs' + str(end).replace('.', '', 1) + '.txt')
            Profile.write_file(wr_text, f_name, cfg.SAVE_RESULTS)
            print(wr_text)
    elif algChoice == 'pls':
        # PLS-GRAANK
        if pTune == 1:
            res_json = pls_grad.parameter_tuning()
            stamp = time.time()
            f_name = str('tune_pls' + str(stamp).replace('.', '', 1) + '.txt')
            Profile.write_file(str(res_json), f_name, cfg.SAVE_RESULTS)
            print(res_json)
        else:
            start = time.time()
            tracemalloc.start()
            res_text = pls_grad.execute(filePath, minSup, numCores, cfg.MAX_ITERATIONS, cfg.MAX_EVALUATIONS, stepVal,
                                        cfg.N_VAR, VISUAL)
            snapshot = tracemalloc.take_snapshot()
            end = time.time()

            wr_text = ("Run-time: " + str(end - start) + " seconds\n")
            wr_text += (Profile.get_quick_mem_use(snapshot) + "\n")
            wr_text += str(res_text)
            f_name = str('res_pls' + str(end).replace('.', '', 1) + '.txt')
            Profile.write_file(wr_text, f_name, cfg.SAVE_RESULTS)
            print(wr_text)
    else:
        print("Invalid Algorithm Choice!")