import numpy as np
from .gp import GI, GP


class Numeric:

    @staticmethod
    def decode_gp(attr_keys, position):
        temp_gp = GP()
        if position is None:
            return temp_gp

        bin_str = bin(int(position))[2:]
        bin_arr = np.array(list(bin_str), dtype=int)

        for i in range(bin_arr.size):
            bin_val = bin_arr[i]
            if bin_val == 1:
                gi = GI.parse_gi(attr_keys[i])
                if not temp_gp.contains_attr(gi):
                    temp_gp.add_gradual_item(gi)
        return temp_gp

    @staticmethod
    def cost_func(position, attr_keys, d_set):
        pattern = Numeric.decode_gp(attr_keys, position)
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
        bin_sum = np.sum(temp_bin)
        if bin_sum > 0:
            cost = (1 / bin_sum)
        else:
            cost = 1
        return cost

    @staticmethod
    def apply_bound(x, var_min, var_max):
        x.position = np.maximum(x.position, var_min)
        x.position = np.minimum(x.position, var_max)


class Bitmap:

    @staticmethod
    def cost_func(gene, attr_keys, d_set):
        pattern = Bitmap.decode_gp(attr_keys, gene)
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
        bin_sum = np.sum(temp_bin)
        if bin_sum > 0:
            cost = (1 / bin_sum)
        else:
            cost = 1
        return cost

    @staticmethod
    def decode_gp(attr_keys, gene):
        temp_gp = GP()
        if gene is None:
            return temp_gp
        for a in range(gene.shape[0]):
            gi = None
            if gene[a][0] > gene[a][1]:
                gi = GI.parse_gi(attr_keys[a][0])
            elif gene[a][1] > gene[a][0]:
                gi = GI.parse_gi(attr_keys[a][1])
            if not (gi is None) and (not temp_gp.contains_attr(gi)):
                temp_gp.add_gradual_item(gi)
        return temp_gp

    @staticmethod
    def decode_encoding(gene):
        if gene is None:
            return -1

        for i in range(gene.shape[0]):
            if gene[i][0] > gene[i][1]:
                gene[i][0] = 1
                gene[i][1] = 0
            elif gene[i][0] < gene[i][1]:
                gene[i][0] = 0
                gene[i][1] = 1
            elif (gene[i][0] == gene[i][1]) and gene[i][0] != 0:
                gene[i][0] = 1
                gene[i][1] = 1
            else:
                gene[i][0] = 0
                gene[i][1] = 0
        gene = np.array(gene).ravel()
        val = gene.dot(2 ** np.arange(gene.size)[::-1])
        return val

    @staticmethod
    def build_gp_gene(attr_keys):
        a = attr_keys
        temp_gene = np.random.choice(a=[0, 1], size=(len(a), 2))
        return temp_gene
