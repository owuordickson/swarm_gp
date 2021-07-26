# -*- coding: utf-8 -*-
import config as cfg
import time
import tracemalloc
from src.pkg_algorithms.prs_grad import init
from src.pkg_algorithms.shared.profile import Profile

# res = pkg_init(cfg.DATASET, cfg.MIN_SUPPORT, cfg.CPU_CORES)
# print(res)

start = time.time()
tracemalloc.start()
res_text = init(cfg.DATASET, cfg.MIN_SUPPORT, cfg.CPU_CORES)
snapshot = tracemalloc.take_snapshot()
end = time.time()

wr_text = ("Run-time: " + str(end - start) + " seconds\n")
wr_text += (Profile.get_quick_mem_use(snapshot) + "\n")
wr_text += str(res_text)
f_name = str('res_prs' + str(end).replace('.', '', 1) + '.txt')
Profile.write_file(wr_text, f_name)
print(wr_text)