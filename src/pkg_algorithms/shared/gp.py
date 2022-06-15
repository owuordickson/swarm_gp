# -*- coding: utf-8 -*-
"""
@author: "Dickson OWUOR"
@credits: "Anne LAURENT and Joseph ORERO"
@version: "4.8"
@email: "owuordickson@gmail.com"
@created: "20 May 2020"
@modified: "10 Mar 2021"

GI: Gradual Item (0, +)
GP: Gradual Pattern {(0, +), (1, -), (3, +)}
TGP: Temporal Gradual Pattern

"""
import numpy as np


class GI:

    def __init__(self, attr_col, symbol):
        self.attribute_col = attr_col
        self.symbol = symbol
        self.gradual_item = np.array((attr_col, symbol), dtype='i, S1')
        self.tuple = tuple([attr_col, symbol])
        self.rank_sum = 0

    def inv(self):
        if self.symbol == '+':
            # temp = tuple([self.attribute_col, '-'])
            temp = np.array((self.attribute_col, '-'), dtype='i, S1')
        elif self.symbol == '-':
            # temp = tuple([self.attribute_col, '+'])
            temp = np.array((self.attribute_col, '+'), dtype='i, S1')
        else:
            temp = np.array((self.attribute_col, 'x'), dtype='i, S1')
        return temp

    def as_integer(self):
        if self.symbol == '+':
            temp = [self.attribute_col, 1]
        elif self.symbol == '-':
            temp = [self.attribute_col, -1]
        else:
            temp = [self.attribute_col, 0]
        return temp

    def as_string(self):
        if self.symbol == '+':
            temp = str(self.attribute_col) + '_pos'
        elif self.symbol == '-':
            temp = str(self.attribute_col) + '_neg'
        else:
            temp = str(self.attribute_col) + '_inv'
        return temp

    def to_string(self):
        return str(self.attribute_col) + self.symbol

    def is_decrement(self):
        if self.symbol == '-':
            return True
        else:
            return False

    @staticmethod
    def parse_gi(gi_str):
        txt = gi_str.split('_')
        attr_col = int(txt[0])
        if txt[1] == 'neg':
            symbol = '-'
        else:
            symbol = '+'
        return GI(attr_col, symbol)


class GP:

    def __init__(self):
        self.gradual_items = list()
        self.support = 0

    def set_support(self, support):
        self.support = support

    def add_gradual_item(self, item):
        if item.symbol == '-' or item.symbol == '+':
            self.gradual_items.append(item)
        else:
            pass

    def get_pattern(self):
        pattern = list()
        for item in self.gradual_items:
            pattern.append(item.gradual_item.tolist())
        return pattern

    def get_np_pattern(self):
        pattern = []
        for item in self.gradual_items:
            pattern.append(item.gradual_item)
        return np.array(pattern)

    def get_tuples(self):
        pattern = list()
        for gi in self.gradual_items:
            temp = tuple([gi.attribute_col, gi.symbol])
            pattern.append(temp)
        return pattern

    def get_attributes(self):
        attrs = list()
        syms = list()
        for item in self.gradual_items:
            gi = item.as_integer()
            attrs.append(gi[0])
            syms.append(gi[1])
        return attrs, syms

    def get_index(self, gi):
        for i in range(len(self.gradual_items)):
            gi_obj = self.gradual_items[i]
            if (gi.symbol == gi_obj.symbol) and (gi.attribute_col == gi_obj.attribute_col):
                return i
        return -1

    def inv_pattern(self):
        pattern = list()
        for gi in self.gradual_items:
            pattern.append(gi.inv().tolist())
        return pattern

    def contains(self, gi):
        if gi is None:
            return False
        if gi in self.gradual_items:
            return True
        return False

    def contains_attr(self, gi):
        if gi is None:
            return False
        for gi_obj in self.gradual_items:
            if gi.attribute_col == gi_obj.attribute_col:
                return True
        return False

    def to_string(self):
        pattern = list()
        for item in self.gradual_items:
            pattern.append(item.to_string())
        return pattern

    def to_dict(self):
        gi_dict = {}
        for gi in self.gradual_items:
            gi_dict.update({gi.as_string(): 0})
        return gi_dict


def validate_gp(d_set, pattern):

    # pattern = [('2', '+'), ('4', '+')]
    min_supp = d_set.thd_supp
    n = d_set.attr_size
    gen_pattern = GP()
    bin_arr = np.array([])

    for gi in pattern.gradual_items:
        arg = np.argwhere(np.isin(d_set.valid_bins[:, 0], gi.gradual_item))
        if len(arg) > 0:
            i = arg[0][0]
            valid_bin = d_set.valid_bins[i]
            if bin_arr.size <= 0:
                bin_arr = np.array([valid_bin[1], valid_bin[1]])
                gen_pattern.add_gradual_item(gi)
            else:
                bin_arr[1] = valid_bin[1].copy()
                temp_bin = np.multiply(bin_arr[0], bin_arr[1])
                supp = float(np.sum(temp_bin)) / float(n * (n - 1.0) / 2.0)
                if supp >= min_supp:
                    bin_arr[0] = temp_bin.copy()
                    gen_pattern.add_gradual_item(gi)
                    gen_pattern.set_support(supp)
    if len(gen_pattern.gradual_items) <= 1:
        return pattern
    else:
        return gen_pattern


def check_anti_monotony(lst_p, pattern, subset=True):
    result = False
    if subset:
        for pat in lst_p:
            result1 = set(pattern.get_pattern()).issubset(set(pat.get_pattern()))
            result2 = set(pattern.inv_pattern()).issubset(set(pat.get_pattern()))
            if result1 or result2:
                result = True
                break
    else:
        for pat in lst_p:
            result1 = set(pattern.get_pattern()).issuperset(set(pat.get_pattern()))
            result2 = set(pattern.inv_pattern()).issuperset(set(pat.get_pattern()))
            if result1 or result2:
                result = True
                break
    return result


def is_duplicate(pattern, lst_winners):
    for pat in lst_winners:
        if set(pattern.get_pattern()) == set(pat.get_pattern()) or \
                set(pattern.inv_pattern()) == set(pat.get_pattern()):
            return True
    return False
