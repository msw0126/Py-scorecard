# -*- coding:utf-8 -*-
import operator
import pandas as pd
from Act.ActAudit import act_data_audit


def import_test_data(data, data_type_dict, id_varname, target_varname = None, encoding="UTF-8", miss_symbols="NULL"):
    '''
    Test Data Importing
    Authors: Tao Yang, ShuaiWei.Meng

    Description:
        Imports the raw test data using the data dictionary.

    :param data: data frame of the test data or name of the .csv file of the test data.
    :param data_type_dict: dict of the data dictionary.
    :param id_varname: name of the ID variable of the test data.
    :param target_varname: name of the target variable of the test data. If NULL, then the target variable is not included in the test data.
    :param encoding: name of the encoding of the test data: "unknown", "UTF-8" and "Latin-1".
    :param miss_symbols: character vector of strings which are to be interpreted as missing values (default: "NA", "N/A", "null", "NULL", "?").
    :return: df - data frame of the test data.
            id - vector of the ID variable of the test data.

    Details:
    The requirements of test data:
        1 The first row specifies the names of all the variables.
        2 Each row corresponds to a record and each column corresponds to a variable.
        3 The types of the variables can only include 'numeric', 'factor'.
        4 If the target variable is included, then the target variable must be a binary factor with factor levels of (0,1) and no missing values are allowed in the target variable.
        5 The ID variable must be included in the test data.
        6 The test records cannot have duplicated IDs.
        7 The missing values are represented by miss.symbols (default: "NA", "N/A", "null", "NULL", "?").
        8 The variable names cannot contain the symbol of "-".

    Algorithm procedures:
        1 Imports the test data from a data frame or a .csv file.
        2 Extracts the ID variable.
        3 Extracts the target variable.
        4 Extracts input variables of the test data and adjusts variable types according to data dictionary.
        5 Splits test data into X (input variables) and Y (target variable) (if Y exists).
    Examples:
        df, id = import_test_data("./data/train.csv", data_type_dict, target_varname="Target", id_varname="id")
    '''

    # Error handlings
    if len(data_type_dict.keys()) <= 2:
        print("The number of rows of dictionary is too small: {}".format(data_type_dict.iloc[:, 0].size))
        exit()
    if not isinstance(miss_symbols, str):
        print("The class of miss.symbols must be character.")
        exit()
    if encoding not in ["unknown", "UTF-8", "Latin-1"]:
        print("encoding can only be 'unknown', 'UTF-8' or 'Latin-1'.")
        exit()

    # default miss symbols
    miss_symbols_lst = ["NA", "N/A", "null", "NULL", "?"] + [miss_symbols]

    # read csv file
    data = pd.read_csv(data, na_values=miss_symbols_lst, encoding=encoding)

    print("-----------------------")
    print("Test Data Importing")
    print("-----------------------")

    # conduct data audit
    result_tag, audit_result_dict = act_data_audit(data, id_varname, target_varname, data_type_dict)
    if result_tag is False:
        return 'Audit Fail', audit_result_dict, ' '

    # Removes test records with the same ID and select ID
    data = data.drop_duplicates([id_varname])
    id = data[id_varname]

    # Extracts factor variables
    X_factor_data = pd.DataFrame()
    factor_vars = [i[0] for i in data_type_dict.items() if i[1] == 'factor']
    for col in [x for x in factor_vars if target_varname not in x]:
        X_factor_data = pd.concat([X_factor_data, data[col].astype('category')], axis=1)

    # Extracts numeric variables
    numeric_vars = [i[0] for i in data_type_dict.items() if i[1] == 'numeric']
    X_numeric_data = pd.DataFrame(pd.DataFrame(data, columns=numeric_vars).apply(pd.to_numeric))

    X = pd.concat([X_factor_data, X_numeric_data], axis=1)

    if target_varname in list(data.columns):
        Y = data[target_varname]
        df = pd.concat([X, Y], axis=1)
    else:
        df = X

    print("The test data were imported.\n")

    return df, id