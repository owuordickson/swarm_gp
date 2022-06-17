# import so4gp as sgp
from .shared import so4gp as sgp


def execute(f_path, min_supp, cores,  evaporation_factor, max_iteration):
    try:
        if cores > 1:
            num_cores = cores
        else:
            num_cores = sgp.get_num_cores()

        d_set = sgp.DataGP(f_path, min_supp)
        out_json, list_gp = sgp.acogps(f_path, min_supp, evaporation_factor, max_iteration, return_gps=True)

        import json
        out_obj = json.loads(out_json)

        wr_line = "Algorithm: ACO-GRAANK (v4.0)\n"
        wr_line += "No. of (dataset) attributes: " + str(d_set.col_count) + '\n'
        wr_line += "No. of (dataset) tuples: " + str(d_set.row_count) + '\n'
        wr_line += "Evaporation factor: " + str(evaporation_factor) + '\n'

        wr_line += "Number of iterations: " + str(max_iteration) + '\n'

        wr_line += "Minimum support: " + str(min_supp) + '\n'
        wr_line += "Number of cores: " + str(num_cores) + '\n'
        wr_line += "Number of patterns: " + str(len(list_gp)) + '\n'
        wr_line += "Number of invalid patterns: " + str(out_obj["Invalid Count"]) + '\n\n'

        for txt in d_set.titles:
            try:
                wr_line += (str(txt.key) + '. ' + str(txt.value.decode()) + '\n')
            except AttributeError:
                wr_line += (str(txt[0]) + '. ' + str(txt[1].decode()) + '\n')

        wr_line += str("\nFile: " + f_path + '\n')
        wr_line += str("\nPattern : Support" + '\n')

        for gp in list_gp:
            wr_line += (str(gp.to_string()) + ' : ' + str(round(gp.support, 3)) + '\n')

        return wr_line
    except ArithmeticError as error:
        wr_line = "Failed: " + str(error)
        print(error)
        return wr_line
