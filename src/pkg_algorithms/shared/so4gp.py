# -*- coding: utf-8 -*-
"""
@author: Dickson Owuor

@credits: Thomas Runkler, Edmond Menya, and Anne Laurent

@license: MIT

@version: 0.1.7

@email: owuordickson@gmail.com

@created: 21 July 2021

@modified: 17 March 2022

SO4GP
------

**SO4GP** stands for: "Some Optimizations for Gradual Patterns". SO4GP applies optimizations such as swarm intelligence,
HDF5 chunks, SVD and many others in order to improve the efficiency of extracting gradual patterns (GPs).

 A GP is a set of gradual items (GI) and its quality is measured by its computed support value. A GI is a pair (i,v)
 where i is a column and v is a variation symbol: increasing/decreasing. Each column of a data set yields 2 GIs; for
 example, column age yields GI age+ or age-. For example given a data set with 3 columns (age, salary, cars) and 10
 objects. A GP may take the form: {age+, salary-} with a support of 0.8. This implies that 8 out of 10 objects have the
 values of column age 'increasing' and column 'salary' decreasing.

    The classical approach for mining GPs is computationally expensive. This package provides Python algorithm
implementations of several optimization techniques that are applied to the classical approach in order to improve its
computational efficiency. The algorithm implementations include:

        * (Classical) GRAANK algorithm for extracting GPs
        * Ant Colony Optimization algorithm for extracting GPs
        * Genetic Algorithm for extracting GPs
        * Particle Swarm Optimization algorithm for extracting GPs
        * Random Search algorithm for extracting GPs
        * Local Search algorithm for extracting GPs


"""

import csv
from dateutil.parser import parse
import time
import gc
import numpy as np
import json
import multiprocessing as mp
import os
import pandas as pd
import random
from ypstruct import structure

# -------- CONFIGURATION -------------

# Global Swarm Configurations
MIN_SUPPORT = 0.5
MAX_ITERATIONS = 1
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
VELOCITY = 0.9  # higher values helps to move to next number in search space
PERSONAL_COEFF = 0.01
GLOBAL_COEFF = 0.9
TARGET = 1
TARGET_ERROR = 1e-6
N_PARTICLES = 5

# PLS-GRAD Configurations
STEP_SIZE = 0.5

# -------- DATA SET PREPARATION -------------

"""
Changes
-------
1. Fetch all binaries during initialization
2. Replaced loops for fetching binary rank with numpy function
3. Accepts CSV file as data source OR
4. Accepts Pandas DataFrame as data source
5. Cleans data set
"""


class DataGP:
    """Description of class DataGP

    A class for creating data-gp objects. A data-gp object is meant to store all the parameters required by GP
    algorithms to extract gradual patterns (GP). It takes a numeric file (in CSV format) as input and converts it into
    an object whose attributes are used by algorithms to extract GPs.

        A GP is a set of gradual items (GI) and its quality is measured by its computed support value. For example given
    a data set with 3 columns (age, salary, cars) and 10 objects. A GP may take the form: {age+, salary-} with a support
    of 0.8. This implies that 8 out of 10 objects have the values of column age 'increasing' and column 'salary'
    decreasing.

    The class provides the following attributes:
        thd_supp: minimum support threshold

        equal: eq value

        titles: column names of data source

        data: all the objects organized into their respective column

        row_count: number of objects

        col_count: number of all columns

        time_cols: column indices of the columns with data-time objects

        attr_cols: column indices of the columns with numeric values

        valid_bins: valid bitmaps (in the form of ndarray) of all gradual items corresponding to the attr_cols, a bitmap is valid if its computed support is equal or greater than the minimum support threshold

        no_bins: True if all none of the attr_cols yields a valid bitmap

    """

    def __init__(self, data_source, min_sup=MIN_SUPPORT, eq=False):
        """Description of class DataGP


        A class for creating data-gp objects. A data-gp object is meant to store all the parameters required by GP
        algorithms to extract gradual patterns (GP). It takes a numeric file (in CSV format) as input and converts it
        into an object whose attributes are used by algorithms to extract GPs.

        It provides the following attributes:
            thd_supp: minimum support threshold

            equal: eq value

            titles: column names of data source

            data: all the objects organized into their respective column

            row_count: number of objects

            col_count: number of all columns

            time_cols: column indices of the columns with data-time objects

            attr_cols: column indices of the columns with numeric values

            valid_bins: valid bitmaps (in the form of ndarray) of all gradual items corresponding to the attr_cols, a bitmap is valid if its computed support is equal or greater than the minimum support threshold

            net_wins: a net-wins matrix constructed from valid gradual item bitmaps

            no_bins: True if all none of the attr_cols yields a valid bitmap

        :param data_source: [required] a numeric data source, it can either be a 'file in csv format' or a 'Pandas DataFrame'
        :type data_source: str
        :param min_sup: [optional] minimum support threshold, the default is 0.5
        :param eq: encode equal values as gradual, the default is False

        """
        self.thd_supp = min_sup
        """:type thd_supp: float"""
        self.equal = eq
        """:type eq: bool"""
        self.titles, self.data = DataGP.read(data_source)
        """:type titles: ndarray"""
        """:type data: ndarray"""
        self.row_count, self.col_count = self.data.shape
        self.time_cols = self.get_time_cols()
        self.attr_cols = self.get_attr_cols()
        self.valid_bins = np.array([])
        self.no_bins = False
        self.step_name = ''  # For T-GRAANK
        self.attr_size = 0  # For T-GRAANK

    def get_attr_cols(self):
        """
        Returns indices of all columns with non-datetime objects

        :return: ndarray
        """
        all_cols = np.arange(self.col_count)
        attr_cols = np.setdiff1d(all_cols, self.time_cols)
        return attr_cols

    def get_time_cols(self):
        """
        Tests each column's objects for date-time values. Returns indices of all columns with date-time objects

        :return: ndarray
        """
        # Retrieve first column only
        time_cols = list()
        n = self.col_count
        for i in range(n):  # check every column/attribute for time format
            row_data = str(self.data[0][i])
            try:
                time_ok, t_stamp = DataGP.test_time(row_data)
                if time_ok:
                    time_cols.append(i)
            except ValueError:
                continue
        return np.array(time_cols)

    def get_gi_bitmap(self, col):
        if col in self.time_cols:
            raise Exception("Error: " + str(self.titles[col][1].decode()) + " is a date/time column!")
        elif col >= self.col_count:
            raise Exception("Error: Column does not exist!")
        else:
            attr_data = self.data.T
            # n = d_set.row_count
            col_data = np.array(attr_data[col], dtype=float)
            with np.errstate(invalid='ignore'):
                temp_pos = np.where(col_data < col_data[:, np.newaxis], 1, 0)
            return temp_pos

    def init_attributes(self, attr_data=None):
        """
        Generates bitmaps for columns with numeric objects. It only stores (attribute valid_bins) those bitmaps whose
        computed support values are greater or equal to the minimum support threshold value.

        :param attr_data: stepped attribute objects
        :return: void
        """
        # (check) implement parallel multiprocessing
        # 1. Transpose csv array data
        if attr_data is None:
            attr_data = self.data.T
            self.attr_size = self.row_count
        else:
            self.attr_size = len(attr_data[self.attr_cols[0]])

        # 2. Construct and store 1-item_set valid bins
        # execute binary rank to calculate support of pattern
        n = self.attr_size
        valid_bins = list()
        for col in self.attr_cols:
            col_data = np.array(attr_data[col], dtype=float)
            incr = np.array((col, '+'), dtype='i, S1')
            decr = np.array((col, '-'), dtype='i, S1')

            # 2a. Generate 1-itemset gradual items
            with np.errstate(invalid='ignore'):
                if not self.equal:
                    temp_pos = col_data < col_data[:, np.newaxis]
                else:
                    temp_pos = col_data <= col_data[:, np.newaxis]
                    np.fill_diagonal(temp_pos, 0)

                # 2b. Check support of each generated itemset
                supp = float(np.sum(temp_pos)) / float(n * (n - 1.0) / 2.0)
                if supp >= self.thd_supp:
                    valid_bins.append(np.array([incr.tolist(), temp_pos], dtype=object))
                    valid_bins.append(np.array([decr.tolist(), temp_pos.T], dtype=object))
        self.valid_bins = np.array(valid_bins)
        # print(self.valid_bins)
        if len(self.valid_bins) < 3:
            self.no_bins = True
        gc.collect()

    @staticmethod
    def read(data_src):
        """
        Reads all the contents of a file (in CSV format) or a data-frame. Checks if its columns have numeric values. It
        separates its columns headers (titles) from the objects.

        :param data_src:
        :return: title, column objects
        """
        # 1. Retrieve data set from source
        if isinstance(data_src, pd.DataFrame):
            # a. DataFrame source
            # Check column names
            try:
                # Check data type
                _ = data_src.columns.astype(float)

                # Add column values
                data_src.loc[-1] = data_src.columns.to_numpy(dtype=float)  # adding a row
                data_src.index = data_src.index + 1  # shifting index
                data_src.sort_index(inplace=True)

                # Rename column names
                vals = ['col_' + str(k) for k in np.arange(data_src.shape[1])]
                data_src.columns = vals
            except ValueError:
                pass
            except TypeError:
                pass
            print("Data fetched from DataFrame")
            return DataGP.clean_data(data_src)
        else:
            # b. CSV file
            file = str(data_src)
            try:
                with open(file, 'r') as f:
                    dialect = csv.Sniffer().sniff(f.readline(), delimiters=";,' '\t")
                    f.seek(0)
                    reader = csv.reader(f, dialect)
                    raw_data = list(reader)
                    f.close()

                if len(raw_data) <= 1:
                    raise Exception("CSV file read error. File has little or no data")
                else:
                    # print("Data fetched from CSV file")
                    # 2. Get table headers
                    keys = np.arange(len(raw_data[0]))
                    if raw_data[0][0].replace('.', '', 1).isdigit() or raw_data[0][0].isdigit():
                        vals = ['col_' + str(k) for k in keys]
                        header = np.array(vals, dtype='S')
                    else:
                        if raw_data[0][1].replace('.', '', 1).isdigit() or raw_data[0][1].isdigit():
                            vals = ['col_' + str(k) for k in keys]
                            header = np.array(vals, dtype='S')
                        else:
                            header = np.array(raw_data[0], dtype='S')
                            raw_data = np.delete(raw_data, 0, 0)
                    # titles = np.rec.fromarrays((keys, values), names=('key', 'value'))
                    # return titles, np.asarray(raw_data)
                    d_frame = pd.DataFrame(raw_data, columns=header)
                    return DataGP.clean_data(d_frame)
            except Exception as error:
                raise Exception("Error: " + str(error))

    @staticmethod
    def test_time(date_str):
        # add all the possible formats
        try:
            if type(int(date_str)):
                return False, False
        except ValueError:
            try:
                if type(float(date_str)):
                    return False, False
            except ValueError:
                try:
                    date_time = parse(date_str)
                    t_stamp = time.mktime(date_time.timetuple())
                    return True, t_stamp
                except ValueError:
                    raise ValueError('no valid date-time format found')

    @staticmethod
    def clean_data(df):
        # 1. Remove objects with Null values
        df = df.dropna()

        # 2. Remove columns with Strings
        cols_to_remove = []
        for col in df.columns:
            try:
                _ = df[col].astype(float)
            except ValueError:
                # Keep time columns
                try:
                    ok, stamp = DataGP.test_time(str(df[col][0]))
                    if not ok:
                        cols_to_remove.append(col)
                except ValueError:
                    cols_to_remove.append(col)
                pass
            except TypeError:
                cols_to_remove.append(col)
                pass
        # keep only the columns in df that do not contain string
        df = df[[col for col in df.columns if col not in cols_to_remove]]

        # 3. Return titles and data
        if df.empty:
            raise Exception("Data set is empty after cleaning.")

        keys = np.arange(df.shape[1])
        values = np.array(df.columns, dtype='S')
        titles = np.rec.fromarrays((keys, values), names=('key', 'value'))
        # print("Data cleaned")
        return titles, df.values


# -------- OTHER METHODS -----------

def get_num_cores():
    num_cores = get_slurm_cores()
    if not num_cores:
        num_cores = mp.cpu_count()
    return num_cores


def get_slurm_cores():
    try:
        cores = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
        return cores
    except ValueError:
        try:
            str_cores = str(os.environ['SLURM_JOB_CPUS_PER_NODE'])
            temp = str_cores.split('(', 1)
            cpus = int(temp[0])
            str_nodes = temp[1]
            temp = str_nodes.split('x', 1)
            str_temp = str(temp[1]).split(')', 1)
            nodes = int(str_temp[0])
            cores = cpus * nodes
            return cores
        except ValueError:
            return False
    except KeyError:
        return False


def write_file(data, path, wr=True):
    if wr:
        with open(path, 'w') as f:
            f.write(data)
            f.close()
    else:
        pass


# -------- GRADUAL PATTERNS -------------

"""
@created: "20 May 2020"
@modified: "21 Jul 2021"

GI: Gradual Item (0, +)
GP: Gradual Pattern {(0, +), (1, -), (3, +)}

"""


class GI:
    """Description of class GI (Gradual Item)

    A class that is used to create GI objects. A GI is a pair (i,v) where i is a column and v is a variation symbol -
    increasing/decreasing. Each column of a data set yields 2 GIs; for example, column age yields GI age+ or age-.

    This class has the following attributes:
        attribute_col: column index of a data set

        symbol: variation symbol (either + or -)

        gradual_item: the GI in a ndarray format

        tuple: the GI in a tuple format

        rank_sum:

    """

    def __init__(self, attr_col, symbol):
        """Description of class GI (Gradual Item)

        A class that is used to create GI objects. A GI is a pair (i,v) where is a column and v is a variation symbol -
        increasing/decreasing. Each column of a data set yields 2 GIs; for example, column age yields GI age+ or age-.

        This class has the following attributes:
            attribute_col: column index of a data set

            symbol: variation symbol (either + or -)

            gradual_item: the GI in a ndarray format

            tuple: the GI in a tuple format

            rank_sum:

        :param attr_col: column index
        :param symbol: variation symbol (either + or -)
        """
        self.attribute_col = attr_col
        """:type attribute_col: int"""
        self.symbol = symbol
        """:type symbol: str"""
        self.gradual_item = np.array((attr_col, symbol), dtype='i, S1')
        self.tuple = tuple([attr_col, symbol])
        self.rank_sum = 0

    def inv(self):
        """
        Inverts a GI to the opposite variation (i.e., from - to +; or, from + to -)
        :return: inverted GI
        """
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
        """
        Converts variation symbol into an integer (i.e., + to 1; and - to -1)
        :return: GI with an integer variation symbol
        """
        if self.symbol == '+':
            temp = [self.attribute_col, 1]
        elif self.symbol == '-':
            temp = [self.attribute_col, -1]
        else:
            temp = [self.attribute_col, 0]
        return temp

    def as_string(self):
        """
        Stringifies a GI. It converts variation symbol into a string (i.e., + to _pos; and - to _neg)
        :return: GI with a string variation symbol
        """
        if self.symbol == '+':
            temp = str(self.attribute_col) + '_pos'
        elif self.symbol == '-':
            temp = str(self.attribute_col) + '_neg'
        else:
            temp = str(self.attribute_col) + '_inv'
        return temp

    def to_string(self):
        """
        Returns a GI in string format
        :return: string
        """
        return str(self.attribute_col) + self.symbol

    def is_decrement(self):
        """
        Checks if a GI's variation corresponds to decreasing
        :return: True is GI has a decreasing variation, False otherwise
        """
        if self.symbol == '-':
            return True
        else:
            return False

    @staticmethod
    def parse_gi(gi_str):
        """
        Converts a stringified GI into normal GI.
        :param gi_str: stringified GI
        :return: GI
        """
        txt = gi_str.split('_')
        attr_col = int(txt[0])
        if txt[1] == 'neg':
            symbol = '-'
        else:
            symbol = '+'
        return GI(attr_col, symbol)


class GP:
    """Description of class GP (Gradual Pattern)

    A class that is used to create GP objects. a GP object is a set of gradual items (GI) and its quality is measured by
    its computed support value. For example given a data set with 3 columns (age, salary, cars) and 10 objects. A GP may
    take the form: {age+, salary-} with a support of 0.8. This implies that 8 out of 10 objects have the values of
    column age 'increasing' and column 'salary' decreasing.

     The class has the following attributes:
        gradual_items: list if GIs
        support: computed support value as a float

    """

    def __init__(self):
        """Description of class GP (Gradual Pattern)

            A class that is used to create GP objects. a GP object is a set of gradual items (GI) and its quality is measured by
            its computed support value. For example given a data set with 3 columns (age, salary, cars) and 10 objects. A GP may
            take the form: {age+, salary-} with a support of 0.8. This implies that 8 out of 10 objects have the values of
            column age 'increasing' and column 'salary' decreasing.

             The class has the following attributes:
                gradual_items: list if GIs
                support: computed support value as a float

            """
        self.gradual_items = list()
        self.support = 0
        """:type support: float"""

    def set_support(self, support):
        """
        Sets the computed support value of the gradual pattern (GP)
        :param support: support value
        :return: void
        """
        self.support = round(support, 3)

    def add_gradual_item(self, item):
        """
        Adds a gradual item (GI) into the gradual pattern (GP)
        :param item: gradual item
        :return: void
        """
        if item.symbol == '-' or item.symbol == '+':
            self.gradual_items.append(item)
        else:
            pass

    def get_pattern(self):
        """
        Returns the gradual pattern (GP) as a list
        :return: gradual pattern
        """
        pattern = list()
        for item in self.gradual_items:
            pattern.append(item.gradual_item.tolist())
        return pattern

    def get_np_pattern(self):
        """
        Returns a gradual pattern (GP) as a ndarray
        :return: ndarray
        """
        pattern = []
        for item in self.gradual_items:
            pattern.append(item.gradual_item)
        return np.array(pattern)

    def get_tuples(self):
        """
        Returns the gradual pattern (GP) as a list of GI tuples
        :return: list of GI tuples
        """
        pattern = list()
        for gi in self.gradual_items:
            temp = tuple([gi.attribute_col, gi.symbol])
            pattern.append(temp)
        return pattern

    def get_attributes(self):
        """
        Breaks down all the gradual items (GIs) in the gradual pattern into columns and variation symbols and returns
        them as separate variables
        :return: separate columns and variation symbols
        """
        attrs = list()
        syms = list()
        for item in self.gradual_items:
            gi = item.as_integer()
            attrs.append(gi[0])
            syms.append(gi[1])
        return attrs, syms

    def get_index(self, gi):
        """
        Returns the index position of a gradual item in the gradual pattern
        :param gi: gradual item
        :return: index of gradual item
        """
        for i in range(len(self.gradual_items)):
            gi_obj = self.gradual_items[i]
            if (gi.symbol == gi_obj.symbol) and (gi.attribute_col == gi_obj.attribute_col):
                return i
        return -1

    def inv_pattern(self):
        """
        Inverts all the variation symbols of all the gradual items (GIs) in the gradual pattern (GP)
        :return: inverted GP
        """
        pattern = list()
        for gi in self.gradual_items:
            pattern.append(gi.inv().tolist())
        return pattern

    def contains(self, gi):
        """
        Checks if a gradual item (GI) is a member of a gradual pattern (GP)
        :param gi: gradual item
        :return: True if it is a member, otherwise False
        """
        if gi is None:
            return False
        if gi in self.gradual_items:
            return True
        return False

    def contains_strict(self, gi):
        """
        Strictly checks if a gradual item (GI) is a member of a gradual pattern (GP)
        :param gi: gradual item
        :return: True if it is a member, otherwise False
        """
        if gi is None:
            return False
        for gi_obj in self.gradual_items:
            if (gi.attribute_col == gi_obj.attribute_col) and (gi.symbol == gi_obj.symbol):
                return True
        return False

    def contains_attr(self, gi):
        """
        Checks is any gradual item (GI) in the gradual pattern (GP) is composed of the column
        :param gi: gradual item
        :return: True if column exists, False otherwise
        """
        if gi is None:
            return False
        for gi_obj in self.gradual_items:
            if gi.attribute_col == gi_obj.attribute_col:
                return True
        return False

    def to_string(self):
        """
        Returns the GP in string format
        :return: string
        """
        pattern = list()
        for item in self.gradual_items:
            pattern.append(item.to_string())
        return pattern

    def to_dict(self):
        """
        Returns the GP as a dictionary
        :return: dict
        """
        gi_dict = {}
        for gi in self.gradual_items:
            gi_dict.update({gi.as_string(): 0})
        return gi_dict

    def print(self, columns):
        """
        Returns patterns with actual column names
        :param columns: Columns names
        :return: GP with actual column names
        """
        pattern = list()
        for item in self.gradual_items:
            col_title = columns[item.attribute_col]
            try:
                col = str(col_title.value.decode())
            except AttributeError:
                col = str(col_title[1].decode())
            pat = str(col + item.symbol)
            pattern.append(pat)  # (item.to_string())
        return [pattern, self.support]


class TimeLag:

    def __init__(self, tstamp=0, supp=0):
        self.timestamp = tstamp
        self.support = round(supp, 3)
        self.sign = self.get_sign()
        if tstamp == 0:
            self.time_lag = np.array([])
            self.valid = False
        else:
            self.time_lag = np.array(self.format_time())
            self.valid = True

    def get_sign(self):
        if self.timestamp < 0:
            sign = "-"
        else:
            sign = "+"
        return sign

    def format_time(self):
        stamp_in_seconds = abs(self.timestamp)
        years = stamp_in_seconds / 3.154e+7
        months = stamp_in_seconds / 2.628e+6
        weeks = stamp_in_seconds / 604800
        days = stamp_in_seconds / 86400
        hours = stamp_in_seconds / 3600
        minutes = stamp_in_seconds / 60
        if int(years) <= 0:
            if int(months) <= 0:
                if int(weeks) <= 0:
                    if int(days) <= 0:
                        if int(hours) <= 0:
                            if int(minutes) <= 0:
                                return [round(stamp_in_seconds, 0), "seconds"]
                            else:
                                return [round(minutes, 0), "minutes"]
                        else:
                            return [round(hours, 0), "hours"]
                    else:
                        return [round(days, 0), "days"]
                else:
                    return [round(weeks, 0), "weeks"]
            else:
                return [round(months, 0), "months"]
        else:
            return [round(years, 0), "years"]

    def to_string(self):
        if len(self.time_lag) > 0:
            txt = ("~ " + self.sign + str(self.time_lag[0]) + " " + str(self.time_lag[1])
                   + " : " + str(self.support))
        else:
            txt = "No time lag found!"
        return txt


# --------- GRAANK ---------------------

"""
CHANGES:
1. Removed T-GRAANK modification
"""


def inv(g_item):
    if g_item[1] == '+':
        temp = tuple([g_item[0], '-'])
    else:
        temp = tuple([g_item[0], '+'])
    return temp


def genapri(R, sup, n):
    """
    Generates Apriori GP candidates
    :param R: attributes
    :param sup: minimum support threshold
    :param n: number of objects
    :return:
    """
    invalid_count = 0
    res = []
    I = []
    if len(R) < 2:
        return []
    try:
        Ck = [{x[0]} for x in R]
    except TypeError:
        Ck = [set(x[0]) for x in R]

    for i in range(len(R) - 1):
        for j in range(i + 1, len(R)):
            try:
                R_i = {R[i][0]}
                R_j = {R[j][0]}
                R_o = {R[0][0]}
            except TypeError:
                R_i = set(R[i][0])
                R_j = set(R[j][0])
                R_o = set(R[0][0])
            temp = R_i | R_j
            invtemp = {inv(x) for x in temp}
            if (len(temp) == len(R_o) + 1) and (not (I != [] and temp in I)) \
                    and (not (I != [] and invtemp in I)):
                test = 1
                for k in temp:
                    try:
                        k_set = {k}
                    except TypeError:
                        k_set = set(k)
                    temp2 = temp - k_set
                    invtemp2 = {inv(x) for x in temp2}
                    if not temp2 in Ck and not invtemp2 in Ck:
                        test = 0
                        break
                if test == 1:
                    m = R[i][1] * R[j][1]
                    t = float(np.sum(m)) / float(n * (n - 1.0) / 2.0)
                    if t > sup:
                        res.append([temp, m])
                    else:
                        invalid_count += 1
                I.append(temp)
                gc.collect()
    return res, invalid_count


def graank(f_path=None, min_sup=MIN_SUPPORT, eq=False, return_gps=False):
    """
    Extract gradual patterns (GPs) from a numeric data source using the GRAANK approach (proposed in a published
    research paper by Anne Laurent).

     A GP is a set of gradual items (GI) and its quality is measured by its computed support value. For example given a
     data set with 3 columns (age, salary, cars) and 10 objects. A GP may take the form: {age+, salary-} with a support
     of 0.8. This implies that 8 out of 10 objects have the values of column age 'increasing' and column 'salary'
     decreasing.

    :param f_path: [required] a numeric data source, it can either be a 'file in csv format' or a 'Pandas DataFrame'
    :param min_sup: [optional] minimum support threshold, the default is 0.5
    :param eq: [optional] encode equal values as gradual, the default is False
    :param return_gps: [optional] additionally return object GPs, the default is False. If set to True, the method returns 2 items: JSON object, GP list
    :return: JSON object
    """

    d_set = DataGP(f_path, min_sup, eq)
    """:type d_set: DataGP"""
    d_set.init_attributes()

    patterns = []
    """:type patterns: GP list"""
    str_winner_gps = []
    n = d_set.attr_size
    valid_bins = d_set.valid_bins

    invalid_count = 0
    while len(valid_bins) > 0:
        valid_bins, inv_count = genapri(valid_bins, min_sup, n)
        invalid_count += inv_count
        i = 0
        while i < len(valid_bins) and valid_bins != []:
            gi_tuple = valid_bins[i][0]
            bin_data = valid_bins[i][1]
            sup = float(np.sum(np.array(bin_data))) / float(n * (n - 1.0) / 2.0)
            if sup < min_sup:
                del valid_bins[i]
                invalid_count += 1
            else:
                z = 0
                while z < (len(patterns) - 1):
                    if set(patterns[z].get_pattern()).issubset(set(gi_tuple)):
                        del patterns[z]
                    else:
                        z = z + 1

                gp = GP()
                """:type gp: GP"""
                for obj in valid_bins[i][0]:
                    gi = GI(obj[0], obj[1].decode())
                    """:type gi: GI"""
                    gp.add_gradual_item(gi)
                gp.set_support(sup)
                patterns.append(gp)
                str_winner_gps.append(gp.print(d_set.titles))
                i += 1
    # Output
    out = json.dumps({"Algorithm": "GRAANK", "Patterns": str_winner_gps, "Invalid Count": invalid_count})
    """:type out: object"""
    if return_gps:
        return out, patterns
    else:
        return out


# -------- ACO-GRAD -------------

"""
CHANGES:
1. generates distance matrix (d_matrix)
2. uses plain methods
"""


def gend(valid_bins):
    """
    Generates the distance matrix (d)
    :param valid_bins: valid GP bitmaps (whose computed support is greater than the minimum support threshold)
    :return: distance matrix (d) and attribute keys
    """
    v_bins = valid_bins
    # 1. Fetch valid bins group
    attr_keys = [GI(x[0], x[1].decode()).as_string() for x in v_bins[:, 0]]

    # 2. Initialize an empty d-matrix
    n = len(attr_keys)
    d = np.zeros((n, n), dtype=np.dtype('i8'))  # cumulative sum of all segments
    for i in range(n):
        for j in range(n):
            if GI.parse_gi(attr_keys[i]).attribute_col == GI.parse_gi(attr_keys[j]).attribute_col:
                # Ignore similar attributes (+ or/and -)
                continue
            else:
                bin_1 = v_bins[i][1]
                bin_2 = v_bins[j][1]
                # Cumulative sum of all segments for 2x2 (all attributes) gradual items
                d[i][j] += np.sum(np.multiply(bin_1, bin_2))
    # print(d)
    return d, attr_keys


def acogps(f_path, min_supp=MIN_SUPPORT, evaporation_factor=EVAPORATION_FACTOR,
           max_iteration=MAX_ITERATIONS, return_gps=False):
    """
    Extract gradual patterns (GPs) from a numeric data source using the Ant Colony Optimization (ACO-GRAD) approach
    (proposed in a published research paper by Dickson Owuor). A GP is a set of gradual items (GI) and its quality is
    measured by its computed support value. For example given a data set with 3 columns (age, salary, cars) and 10
    objects. A GP may take the form: {age+, salary-} with a support of 0.8. This implies that 8 out of 10 objects have
    the values of column age 'increasing' and column 'salary' decreasing.

     In this approach, it is assumed that every column can be converted into gradual item (GI). If the GI is valid (i.e.
     its computed support is greater than the minimum support threshold) then it is either increasing or decreasing (+
     or -), otherwise it is irrelevant (x). Therefore, a pheromone matrix is built using the number of columns and the
     possible variations (increasing, decreasing, irrelevant) or (+, -, x). The algorithm starts by randomly generating
     GP candidates using the pheromone matrix, each candidate is validated by confirming that its computed support is
     greater or equal to the minimum support threshold. The valid GPs are used to update the pheromone levels and better
     candidates are generated.

    :param f_path: [required] a numeric data source, it can either be a 'file in csv format' or a 'Pandas DataFrame'
    :param min_supp: [optional] minimum support threshold, the default is 0.5
    :param evaporation_factor: [optional] evaporation factor default = 0.5
    :param max_iteration: [optional] maximum iterations default = 1
    :param return_gps: [optional] additionally return object GPs, the default is False. If set to True, the method returns 2 items: JSON object, GP list
    :return: JSON object
    """
    # 0. Initialize and prepare data set
    d_set = DataGP(f_path, min_supp)
    """:type d_set: DataGP"""
    d_set.init_attributes()
    # attr_index = d_set.attr_cols
    # e_factor = evaporation_factor
    d, attr_keys = gend(d_set.valid_bins)  # distance matrix (d) & attributes corresponding to d

    a = d_set.attr_size
    winner_gps = list()  # subsets
    loser_gps = list()  # supersets
    str_winner_gps = list()  # subsets
    repeated = 0
    it_count = 0
    counter = 0

    if d_set.no_bins:
        return []

    # 1. Remove d[i][j] < frequency-count of min_supp
    fr_count = ((min_supp * a * (a - 1)) / 2)
    d[d < fr_count] = 0

    # 3. Initialize pheromones (p_matrix)
    pheromones = np.ones(d.shape, dtype=float)

    invalid_count = 0
    # 4. Iterations for ACO
    # while repeated < 1:
    while counter < max_iteration:
        rand_gp, pheromones = genaco(attr_keys, d, pheromones, evaporation_factor)
        if len(rand_gp.gradual_items) > 1:
            # print(rand_gp.get_pattern())
            exits = isduplicate(rand_gp, winner_gps, loser_gps)
            if not exits:
                repeated = 0
                # check for anti-monotony
                is_super = amcheck(loser_gps, rand_gp, subset=False)
                is_sub = amcheck(winner_gps, rand_gp, subset=True)
                if is_super or is_sub:
                    continue
                gen_gp = validategp(d_set, rand_gp)
                """:type gen_gp: GP"""
                is_present = isduplicate(gen_gp, winner_gps, loser_gps)
                is_sub = amcheck(winner_gps, gen_gp, subset=True)
                if is_present or is_sub:
                    repeated += 1
                else:
                    if gen_gp.support >= min_supp:
                        pheromones = update_pheromones(attr_keys, gen_gp, pheromones)
                        winner_gps.append(gen_gp)
                        str_winner_gps.append(gen_gp.print(d_set.titles))
                    else:
                        loser_gps.append(gen_gp)
                        invalid_count += 1
                if set(gen_gp.get_pattern()) != set(rand_gp.get_pattern()):
                    loser_gps.append(rand_gp)
            else:
                repeated += 1
        else:
            invalid_count += 1
        it_count += 1
        if max_iteration == 1:
            counter = repeated
        else:
            counter = it_count
    # Output
    out = json.dumps({"Algorithm": "ACO-GRAD", "Best Patterns": str_winner_gps, "Invalid Count": invalid_count,
                      "Iterations": it_count})
    """:type out: object"""
    if return_gps:
        return out, winner_gps
    else:
        return out


def genaco(attr_keys, d, p_matrix, e_factor):
    v_matrix = d
    pattern = GP()
    ":type pattern: GP"

    # 1. Generate gradual items with the highest pheromone and visibility
    m = p_matrix.shape[0]
    for i in range(m):
        combine_feature = np.multiply(v_matrix[i], p_matrix[i])
        total = np.sum(combine_feature)
        with np.errstate(divide='ignore', invalid='ignore'):
            probability = combine_feature / total
        cum_prob = np.cumsum(probability)
        r = np.random.random_sample()
        try:
            j = np.nonzero(cum_prob > r)[0][0]
            gi = GI.parse_gi(attr_keys[j])
            """:type gi: GI"""
            if not pattern.contains_attr(gi):
                pattern.add_gradual_item(gi)
        except IndexError:
            continue

    # 2. Evaporate pheromones by factor e
    p_matrix = (1 - e_factor) * p_matrix
    return pattern, p_matrix


def update_pheromones(attr_keys, pattern, p_matrix):
    """
    Updates the pheromone level of the pheromone matrix

    :param attr_keys: attribute keys
    :param pattern: pattern used to update values
    :param p_matrix: an existing pheromone matrix
    :return: updated pheromone matrix
    """
    idx = [attr_keys.index(x.as_string()) for x in pattern.gradual_items]
    for n in range(len(idx)):
        for m in range(n + 1, len(idx)):
            i = idx[n]
            j = idx[m]
            p_matrix[i][j] += 1
            p_matrix[j][i] += 1
    return p_matrix


def validategp(d_set, pattern):
    """
    Validates a candidate gradual pattern (GP) based on support computation. A GP is invalid if its support value is
    less than the minimum support threshold set by the user.

    :param d_set: Data_GP object
    :param pattern: candidate GP
    :return: a valid GP or an empty GP
    """
    # pattern = [('2', '+'), ('4', '+')]
    min_supp = d_set.thd_supp
    n = d_set.attr_size
    gen_pattern = GP()
    """type gen_pattern: GP"""
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


def amcheck(lst_p, pattern, subset=True):
    """
    Anti-monotonicity check. Checks if a GP is a subset or superset of an already existing GP

    :param lst_p: list of existing GPs
    :param pattern: GP to be checked
    :param subset: check if it is a subset
    :return: True if superset/subset, False otherwise
    """
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


def isduplicate(pattern, lst_winners, lst_losers=None):
    """
    Checks if a pattern is in the list of winner GPs or loser GPs

    :param pattern: a GP
    :param lst_winners: list of GPs
    :param lst_losers: list of GPs
    :return: True if pattern is either list, False otherwise
    """
    if lst_losers is None:
        pass
    else:
        for pat in lst_losers:
            if set(pattern.get_pattern()) == set(pat.get_pattern()) or \
                    set(pattern.inv_pattern()) == set(pat.get_pattern()):
                return True
    for pat in lst_winners:
        if set(pattern.get_pattern()) == set(pat.get_pattern()) or \
                set(pattern.inv_pattern()) == set(pat.get_pattern()):
            return True
    return False


# -------- GA-GRAD ----------

"""
@author: "Dickson Owuor"
@credits: "Thomas Runkler, and Anne Laurent,"
@license: "MIT"
@version: "2.0"
@email: "owuordickson@gmail.com"
@created: "29 April 2021"
@modified: "07 September 2021"
Breath-First Search for gradual patterns using Genetic Algorithm (GA-GRAD).
GA is used to learn gradual pattern candidates.
CHANGES:
1. uses normal functions
2. updated cost function to use Binary Array of GPs
3. uses rank order search space
"""


def gagps(data_src, min_supp=MIN_SUPPORT, max_iteration=MAX_ITERATIONS, n_pop=N_POPULATION, pc=PC,
          gamma=GAMMA, mu=MU, sigma=SIGMA, return_gps=False):
    """

    Extract gradual patterns (GPs) from a numeric data source using the Genetic Algorithm (GA-GRAD) approach (proposed
    in a published research paper by Dickson Owuor). A GP is a set of gradual items (GI) and its quality is measured by
    its computed support value. For example given a data set with 3 columns (age, salary, cars) and 10 objects. A GP may
    take the form: {age+, salary-} with a support of 0.8. This implies that 8 out of 10 objects have the values of
    column age 'increasing' and column 'salary' decreasing.

     In this approach, it is assumed that every GP candidate may be represented as a binary gene (or individual) that
     has a unique position and cost. The cost is derived from the computed support of that candidate, the higher the
     support value the lower the cost. The aim of the algorithm is search through a population of individuals (or
     candidates) and find those with the lowest cost as efficiently as possible.

    :param data_src: [required] a numeric data source, it can either be a 'file in csv format' or a 'Pandas DataFrame'
    :param min_supp: [optional] minimum support threshold, the default is set to 0.5
    :param max_iteration: [optional] maximum iterations, the default is set to 1
    :param n_pop: [optional] initial population, the default is set to 5
    :param pc: [optional] offspring population multiple, the default is set 0.5. This determines how fast the population grows.
    :param gamma: [optional] crossover rate, the default is set to 1
    :param mu: [optional] mutation rate, the default is set to 0.9
    :param sigma: [optional] mutation rate, the default is set to 0.9
    :param return_gps: [optional] additionally return object GPs, the default is False. If set to True, the method returns 2 items: JSON object, GP list
    :return: JSON object
    """

    # Prepare data set
    d_set = DataGP(data_src, min_supp)
    d_set.init_attributes()
    attr_keys = [GI(x[0], x[1].decode()).as_string() for x in d_set.valid_bins[:, 0]]

    if d_set.no_bins:
        return []

    # Problem Information
    # costfxn

    # Parameters
    # pc: Proportion of children (if its 1, then nc == npop
    it_count = 0
    eval_count = 0
    counter = 0
    var_min = 0
    var_max = int(''.join(['1'] * len(attr_keys)), 2)

    nc = int(np.round(pc * n_pop / 2) * 2)  # Number of children. np.round is used to get even number of children

    # Empty Individual Template
    empty_individual = structure()
    empty_individual.position = None
    empty_individual.cost = None

    # Initialize Population
    pop = empty_individual.repeat(n_pop)
    for i in range(n_pop):
        pop[i].position = random.randrange(var_min, var_max)
        pop[i].cost = 1  # costfxn(pop[i].position, attr_keys, d_set)
        # if pop[i].cost < best_sol.cost:
        #    best_sol = pop[i].deepcopy()

    # Best Solution Ever Found
    best_sol = empty_individual.deepcopy()
    best_sol.position = pop[0].position
    best_sol.cost = costfxn(best_sol.position, attr_keys, d_set)

    # Best Cost of Iteration
    best_costs = np.empty(max_iteration)
    best_patterns = list()
    str_best_gps = list()
    str_iter = ''
    str_eval = ''

    repeated = 0
    while counter < max_iteration:
        # while eval_count < max_evaluations:
        # while repeated < 1:

        c_pop = []  # Children population
        for _ in range(nc // 2):
            # Select Parents
            q = np.random.permutation(n_pop)
            p1 = pop[q[0]]
            p2 = pop[q[1]]

            # a. Perform Crossover
            c1, c2 = crossover(p1, p2, gamma)

            # Apply Bound
            apply_bound(c1, var_min, var_max)
            apply_bound(c2, var_min, var_max)

            # Evaluate First Offspring
            c1.cost = costfxn(c1.position, attr_keys, d_set)
            if c1.cost < best_sol.cost:
                best_sol = c1.deepcopy()
            eval_count += 1
            str_eval += "{}: {} \n".format(eval_count, best_sol.cost)

            # Evaluate Second Offspring
            c2.cost = costfxn(c2.position, attr_keys, d_set)
            if c2.cost < best_sol.cost:
                best_sol = c2.deepcopy()
            eval_count += 1
            str_eval += "{}: {} \n".format(eval_count, best_sol.cost)

            # b. Perform Mutation
            c1 = mutate(c1, mu, sigma)
            c2 = mutate(c2, mu, sigma)

            # Apply Bound
            apply_bound(c1, var_min, var_max)
            apply_bound(c2, var_min, var_max)

            # Evaluate First Offspring
            c1.cost = costfxn(c1.position, attr_keys, d_set)
            if c1.cost < best_sol.cost:
                best_sol = c1.deepcopy()
            eval_count += 1
            str_eval += "{}: {} \n".format(eval_count, best_sol.cost)

            # Evaluate Second Offspring
            c2.cost = costfxn(c2.position, attr_keys, d_set)
            if c2.cost < best_sol.cost:
                best_sol = c2.deepcopy()
            eval_count += 1
            str_eval += "{}: {} \n".format(eval_count, best_sol.cost)

            # c. Add Offsprings to c_pop
            c_pop.append(c1)
            c_pop.append(c2)

        # Merge, Sort and Select
        pop += c_pop
        pop = sorted(pop, key=lambda x: x.cost)
        pop = pop[0:n_pop]

        best_gp = validategp(d_set, decodegp(attr_keys, best_sol.position))
        """:type best_gp: GP"""
        is_present = isduplicate(best_gp, best_patterns)
        is_sub = amcheck(best_patterns, best_gp, subset=True)
        if is_present or is_sub:
            repeated += 1
        else:
            if best_gp.support >= min_supp:
                best_patterns.append(best_gp)
                str_best_gps.append(best_gp.print(d_set.titles))
            # else:
            #    best_sol.cost = 1

        try:
            # Show Iteration Information
            # Store Best Cost
            best_costs[it_count] = best_sol.cost
            str_iter += "{}: {} \n".format(it_count, best_sol.cost)
        except IndexError:
            pass
        it_count += 1

        if max_iteration == 1:
            counter = repeated
        else:
            counter = it_count
    # Output
    out = json.dumps({"Algorithm": "GA-GRAD", "Best Patterns": str_best_gps, "Iterations": it_count})
    """:type out: object"""
    if return_gps:
        return out, best_patterns
    else:
        return out


def costfxn(position, attr_keys, d_set):
    """
    Computes the cost of a specific binary position (this position can be decoded into a GP candidate). The lower the
    cost the better the quality of that position.

    :param position: binary position
    :param attr_keys: list of gradual items (GIs) in the format of strings
    :param d_set: DataGP object
    :return: cost value
    """
    pattern = decodegp(attr_keys, position)
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


def crossover(p1, p2, gamma=0.1):
    """
    Crosses over the genes of 2 parents (an individual with a specific position and cost) in order to generate 2
    different offsprings.

    :param p1: parent 1 individual
    :param p2: parent 2 individual
    :param gamma: cross-over rate
    :return: 2 offsprings (children)
    """
    c1 = p1.deepcopy()
    c2 = p2.deepcopy()
    alpha = np.random.uniform(0, gamma, 1)
    c1.position = alpha * p1.position + (1 - alpha) * p2.position
    c2.position = alpha * p2.position + (1 - alpha) * p1.position
    return c1, c2


def mutate(x, mu, sigma):
    """
    Mutates an individual's position in order to create a new and different individual.

    :param x: existing individual
    :param mu: mutation rate 1
    :param sigma: mutation rate 2
    :return: new individual
    """
    y = x.deepcopy()
    str_x = str(int(y.position))
    # flag = np.random.rand(*x.position.shape) <= mu
    # ind = np.argwhere(flag)
    # y.position[ind] += sigma*np.random.rand(*ind.shape)
    flag = np.random.rand(*(len(str_x),)) <= mu
    ind = np.argwhere(flag)
    str_y = "0"
    for i in ind:
        val = float(str_x[i[0]])
        val += sigma * np.random.uniform(0, 1, 1)
        if i[0] == 0:
            str_y = "".join(("", "{}".format(int(val)), str_x[1:]))
        else:
            str_y = "".join((str_x[:i[0] - 1], "{}".format(int(val)), str_x[i[0]:]))
        str_x = str_y
    y.position = int(str_y)
    return y


def apply_bound(x, var_min, var_max):
    """
    Ensures that an individual's position is between a minimum and a maximum value. If not, it moves it to the minimum
    or maximum value.

    :param x: existing individual
    :param var_min: minimum value
    :param var_max: maximum value
    :return: void
    """
    x.position = np.maximum(x.position, var_min)
    x.position = np.minimum(x.position, var_max)


def decodegp(attr_keys, position):
    """
    Converts a binary position into a gradual pattern (GP) candidate.

    :param attr_keys: list of gradual items (GIs) in the format of strings
    :param position: binary position
    :return: GP
    """
    temp_gp = GP()
    """:type temp_gp: GP"""
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


# -------- PSO-GRAD ----------

"""
@author: "Dickson Owuor"
@credits: "Thomas Runkler, and Anne Laurent,"
@license: "MIT"
@version: "2.0"
@email: "owuordickson@gmail.com"
@created: "29 April 2021"
@modified: "07 September 2021"
Breath-First Search for gradual patterns using Particle Swarm Optimization (PSO-GRAANK).
PSO is used to learn gradual pattern candidates.
CHANGES:
1. uses normal functions
2. updated fitness function to use Binary Array of GPs
3. uses rank order search space
"""


def psogps(data_src, min_supp=MIN_SUPPORT, max_iteration=MAX_ITERATIONS, n_particles=N_PARTICLES,
           velocity=VELOCITY, coef_p=PERSONAL_COEFF, coef_g=GLOBAL_COEFF, return_gps=False):
    """
    Extract gradual patterns (GPs) from a numeric data source using the Particle Swarm Optimization Algorithm (PSO-GRAD)
    approach (proposed in a published research paper by Dickson Owuor). A GP is a set of gradual items (GI) and its
    quality is measured by its computed support value. For example given a data set with 3 columns (age, salary, cars)
    and 10 objects. A GP may take the form: {age+, salary-} with a support of 0.8. This implies that 8 out of 10 objects
    have the values of column age 'increasing' and column 'salary' decreasing.

     In this approach, it is assumed that every GP candidate may be represented as a particle that has a unique position
     and fitness. The fitness is derived from the computed support of that candidate, the higher the support value the
     higher the fitness. The aim of the algorithm is search through a population of particles (or candidates) and find
     those with the highest fitness as efficiently as possible.

    :param data_src: [required] a numeric data source, it can either be a 'file in csv format' or a 'Pandas DataFrame'
    :param min_supp: [optional] minimum support threshold, the default is set to 0.5
    :param max_iteration: [optional] maximum iterations, the default is set to 1
    :param n_particles: [optional] initial particle population, the default is set to 5
    :param velocity: [optional] particle velocity, the default is set to 0.9
    :param coef_p: [optional] personal coefficient rate, the default is set to 0.01
    :param coef_g: [optional] global coefficient, the default is set to 0.9
    :param return_gps: [optional] additionally return object GPs, the default is False. If set to True, the method returns 2 items: JSON object, GP list
    :return: JSON object
    """
    # Prepare data set
    d_set = DataGP(data_src, min_supp)
    d_set.init_attributes()
    # self.target = 1
    # self.target_error = 1e-6
    attr_keys = [GI(x[0], x[1].decode()).as_string() for x in d_set.valid_bins[:, 0]]

    if d_set.no_bins:
        return []

    it_count = 0
    eval_count = 0
    counter = 0
    var_min = 0
    var_max = int(''.join(['1'] * len(attr_keys)), 2)

    # Empty particle template
    empty_particle = structure()
    empty_particle.position = None
    empty_particle.fitness = None

    # Initialize Population
    particle_pop = empty_particle.repeat(n_particles)
    for i in range(n_particles):
        particle_pop[i].position = random.randrange(var_min, var_max)
        particle_pop[i].fitness = 1

    pbest_pop = particle_pop.copy()
    gbest_particle = pbest_pop[0]

    # Best particle (ever found)
    best_particle = empty_particle.deepcopy()
    best_particle.position = gbest_particle.position
    best_particle.fitness = costfxn(best_particle.position, attr_keys, d_set)

    velocity_vector = np.ones(n_particles)
    best_fitness_arr = np.empty(max_iteration)
    best_patterns = []
    str_best_gps = list()
    str_iter = ''
    str_eval = ''

    repeated = 0
    while counter < max_iteration:
        # while eval_count < max_evaluations:
        # while repeated < 1:
        for i in range(n_particles):
            # UPDATED
            if particle_pop[i].position < var_min or particle_pop[i].position > var_max:
                particle_pop[i].fitness = 1
            else:
                particle_pop[i].fitness = costfxn(particle_pop[i].position, attr_keys, d_set)
                eval_count += 1
                str_eval += "{}: {} \n".format(eval_count, particle_pop[i].fitness)

            if pbest_pop[i].fitness > particle_pop[i].fitness:
                pbest_pop[i].fitness = particle_pop[i].fitness
                pbest_pop[i].position = particle_pop[i].position

            if gbest_particle.fitness > particle_pop[i].fitness:
                gbest_particle.fitness = particle_pop[i].fitness
                gbest_particle.position = particle_pop[i].position
        # if abs(gbest_fitness_value - self.target) < self.target_error:
        #    break
        if best_particle.fitness > gbest_particle.fitness:
            best_particle = gbest_particle.deepcopy()

        for i in range(n_particles):
            new_velocity = (velocity * velocity_vector[i]) + \
                           (coef_p * random.random()) * (pbest_pop[i].position - particle_pop[i].position) + \
                           (coef_g * random.random()) * (gbest_particle.position - particle_pop[i].position)
            particle_pop[i].position = particle_pop[i].position + new_velocity

        best_gp = validategp(d_set, decodegp(attr_keys, best_particle.position))
        """:type best_gp: GP"""
        is_present = isduplicate(best_gp, best_patterns)
        is_sub = amcheck(best_patterns, best_gp, subset=True)
        if is_present or is_sub:
            repeated += 1
        else:
            if best_gp.support >= min_supp:
                best_patterns.append(best_gp)
                str_best_gps.append(best_gp.print(d_set.titles))
            # else:
            #    best_particle.fitness = 1

        try:
            # Show Iteration Information
            best_fitness_arr[it_count] = best_particle.fitness
            str_iter += "{}: {} \n".format(it_count, best_particle.fitness)
        except IndexError:
            pass
        it_count += 1

        if max_iteration == 1:
            counter = repeated
        else:
            counter = it_count
    # Output
    out = json.dumps({"Algorithm": "PSO-GRAD", "Best Patterns": str_best_gps, "Iterations": it_count})
    """:type out: object"""
    if return_gps:
        return out, best_patterns
    else:
        return out


# -------- PLS-GRAD ----------

"""
@author: "Dickson Owuor"
@credits: "Thomas Runkler, and Anne Laurent,"
@license: "MIT"
@version: "2.0"
@email: "owuordickson@gmail.com"
@created: "26 July 2021"
@modified: "07 September 2021"
Breath-First Search for gradual patterns using Pure Local Search (PLS-GRAD).
PLS is used to learn gradual pattern candidates.
Adopted from: https://machinelearningmastery.com/iterated-local-search-from-scratch-in-python/
CHANGES:
1. Used rank order search space
"""


# hill climbing local search algorithm
def hcgps(data_src, min_supp=MIN_SUPPORT, max_iteration=MAX_ITERATIONS, step_size=STEP_SIZE, return_gps=False):
    """
    Extract gradual patterns (GPs) from a numeric data source using the Hill Climbing (Local Search) Algorithm (LS-GRAD)
    approach (proposed in a published research paper by Dickson Owuor). A GP is a set of gradual items (GI) and its
    quality is measured by its computed support value. For example given a data set with 3 columns (age, salary, cars)
    and 10 objects. A GP may take the form: {age+, salary-} with a support of 0.8. This implies that 8 out of 10 objects
    have the values of column age 'increasing' and column 'salary' decreasing.

     In this approach, it is assumed that every GP candidate may be represented as a position that has a cost value
     associated with it. The cost is derived from the computed support of that candidate, the higher the support value
     the lower the cost. The aim of the algorithm is search through group of positions and find those with the lowest
     cost as efficiently as possible.

    :param data_src: [required] a numeric data source, it can either be a 'file in csv format' or a 'Pandas DataFrame'
    :param min_supp: [optional] minimum support threshold, the default is set to 0.5
    :param max_iteration: [optional] maximum iterations, the default is set to 1
    :param step_size: [optional] step size, the default is set to 0.5
    :param return_gps: [optional] additionally return object GPs, the default is False. If set to True, the method returns 2 items: JSON object, GP list
    :return: JSON object
    """
    # Prepare data set
    d_set = DataGP(data_src, min_supp)
    d_set.init_attributes()
    attr_keys = [GI(x[0], x[1].decode()).as_string() for x in d_set.valid_bins[:, 0]]

    if d_set.no_bins:
        return []

    # Parameters
    it_count = 0
    var_min = 0
    counter = 0
    var_max = int(''.join(['1'] * len(attr_keys)), 2)
    eval_count = 0

    # Empty Individual Template
    best_sol = structure()
    candidate = structure()

    # Best Cost of Iteration
    best_costs = np.empty(max_iteration)
    best_patterns = []
    str_best_gps = list()
    str_iter = ''
    str_eval = ''
    repeated = 0

    # generate an initial point
    best_sol.position = None
    # candidate.position = None
    if best_sol.position is None:
        best_sol.position = np.random.uniform(var_min, var_max, N_VAR)
    # evaluate the initial point
    apply_bound(best_sol, var_min, var_max)
    best_sol.cost = costfxn(best_sol.position, attr_keys, d_set)

    # run the hill climb
    while counter < max_iteration:
        # while eval_count < max_evaluations:
        # take a step
        candidate.position = None
        if candidate.position is None:
            candidate.position = best_sol.position + (random.randrange(var_min, var_max) * step_size)
        apply_bound(candidate, var_min, var_max)
        candidate.cost = costfxn(candidate.position, attr_keys, d_set)

        if candidate.cost < best_sol.cost:
            best_sol = candidate.deepcopy()
        eval_count += 1
        str_eval += "{}: {} \n".format(eval_count, best_sol.cost)

        best_gp = validategp(d_set, decodegp(attr_keys, best_sol.position))
        """:type best_gp: GP"""
        is_present = isduplicate(best_gp, best_patterns)
        is_sub = amcheck(best_patterns, best_gp, subset=True)
        if is_present or is_sub:
            repeated += 1
        else:
            if best_gp.support >= min_supp:
                best_patterns.append(best_gp)
                str_best_gps.append(best_gp.print(d_set.titles))

        try:
            # Show Iteration Information
            # Store Best Cost
            best_costs[it_count] = best_sol.cost
            str_iter += "{}: {} \n".format(it_count, best_sol.cost)
        except IndexError:
            pass
        it_count += 1

        if max_iteration == 1:
            counter = repeated
        else:
            counter = it_count
    # Output
    out = json.dumps({"Algorithm": "LS-GRAD", "Best Patterns": str_best_gps, "Iterations": it_count})
    """:type out: object"""
    if return_gps:
        return out, best_patterns
    else:
        return out


# -------- PRS-GRAD ----------

"""
@author: "Dickson Owuor"
@credits: "Thomas Runkler, and Anne Laurent,"
@license: "MIT"
@version: "2.0"
@email: "owuordickson@gmail.com"
@created: "26 July 2021"
@modified: "07 September 2021"
Breath-First Search for gradual patterns using Pure Random Search (PRS-GRAD).
PRS is used to learn gradual pattern candidates.
Adopted: https://medium.com/analytics-vidhya/how-does-random-search-algorithm-work-python-implementation-b69e779656d6
CHANGES:
1. Uses rank-order search space
"""


def rsgps(data_src, min_supp=MIN_SUPPORT, max_iteration=MAX_ITERATIONS, return_gps=False):
    """
    Extract gradual patterns (GPs) from a numeric data source using the Random Search Algorithm (LS-GRAD)
    approach (proposed in a published research paper by Dickson Owuor). A GP is a set of gradual items (GI) and its
    quality is measured by its computed support value. For example given a data set with 3 columns (age, salary, cars)
    and 10 objects. A GP may take the form: {age+, salary-} with a support of 0.8. This implies that 8 out of 10 objects
    have the values of column age 'increasing' and column 'salary' decreasing.

     In this approach, it is assumed that every GP candidate may be represented as a position that has a cost value
     associated with it. The cost is derived from the computed support of that candidate, the higher the support value
     the lower the cost. The aim of the algorithm is search through group of positions and find those with the lowest
     cost as efficiently as possible.

    :param data_src: [required] a numeric data source, it can either be a 'file in csv format' or a 'Pandas DataFrame'
    :param min_supp: [optional] minimum support threshold, the default is set to 0.5
    :param max_iteration: [optional] maximum iterations, the default is set to 1
    :param return_gps: [optional] additionally return object GPs, the default is False. If set to True, the method returns 2 items: JSON object, GP list
    :return: JSON object
    """
    # Prepare data set
    d_set = DataGP(data_src, min_supp)
    d_set.init_attributes()
    attr_keys = [GI(x[0], x[1].decode()).as_string() for x in d_set.valid_bins[:, 0]]

    if d_set.no_bins:
        return []

    # Parameters
    it_count = 0
    counter = 0
    var_min = 0
    var_max = int(''.join(['1'] * len(attr_keys)), 2)
    eval_count = 0

    # Empty Individual Template
    candidate = structure()
    candidate.position = None
    candidate.cost = float('inf')

    # INITIALIZE
    best_sol = candidate.deepcopy()
    best_sol.position = np.random.uniform(var_min, var_max, N_VAR)
    best_sol.cost = costfxn(best_sol.position, attr_keys, d_set)

    # Best Cost of Iteration
    best_costs = np.empty(max_iteration)
    best_patterns = []
    str_best_gps = list()
    str_iter = ''
    str_eval = ''

    repeated = 0
    while counter < max_iteration:
        # while eval_count < max_evaluations:

        candidate.position = ((var_min + random.random()) * (var_max - var_min))
        apply_bound(candidate, var_min, var_max)
        candidate.cost = costfxn(candidate.position, attr_keys, d_set)

        if candidate.cost < best_sol.cost:
            best_sol = candidate.deepcopy()
        eval_count += 1
        str_eval += "{}: {} \n".format(eval_count, best_sol.cost)

        best_gp = validategp(d_set, decodegp(attr_keys, best_sol.position))
        """:type best_gp: GP"""
        is_present = isduplicate(best_gp, best_patterns)
        is_sub = amcheck(best_patterns, best_gp, subset=True)
        if is_present or is_sub:
            repeated += 1
        else:
            if best_gp.support >= min_supp:
                best_patterns.append(best_gp)
                str_best_gps.append(best_gp.print(d_set.titles))
            # else:
            #    best_sol.cost = 1

        try:
            # Show Iteration Information
            # Store Best Cost
            best_costs[it_count] = best_sol.cost
            str_iter += "{}: {} \n".format(it_count, best_sol.cost)
        except IndexError:
            pass
        it_count += 1

        if max_iteration == 1:
            counter = repeated
        else:
            counter = it_count
    # Output
    out = json.dumps({"Algorithm": "RS-GRAD", "Best Patterns": str_best_gps, "Iterations": it_count})
    """:type out: object"""
    if return_gps:
        return out, best_patterns
    else:
        return out
