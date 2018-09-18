# -*- coding:utf-8 -*-

import operator
import pandas as pd
from Learn.DataAudit import explore_data_audit


def import_training_data(data, dictionary, explore_conf_dict, miss_symbols = "NULL", encoding="UTF-8"):
    '''
    Training Data Importing
    Authors: ShuaiWei Meng, Xianda Wu

    Description:
        Imports the raw training data using the data dictionary.

    :param data: data frame of the training data or name of the .csv file of the training data.
    :param dictionary: dict of the data dictionary.
    :param explore_conf_dict: dict of thr explore parameter
    :param miss_symbols: character vector of strings which are to be interpreted as missing values (default: "NA", "N/A", "null", "NULL", "?").
    :param encoding: name of the encoding of the training data and data dictionary: "unknown", "UTF-8" and "Latin-1".
    :return: df - data frame of the training data.
             data_type_dict - dict of the data dictionary.

    Details:
        The requirements of training data:
        1 The first row specifies the names of all the variables.
        2 Each row corresponds to a sample and each column corresponds to a variable.
        3 The types of the variables can only include 'numeric', 'factor'.
        4 The target variable must be included in the training data The target variable must be a binary factor where the majority class is coded as 0 and the minority as 1, and no missing values are allowed in the target variable.
        5 Number of samples >= 100.
        6 Number of variables >= 2.
        7 The missing values are represented by "NA", "N/A", "null", "NULL", "?".
        8 The variable names cannot contain the symbol of "-".

        The requirements of the data dictionary:
        1 Data dictionary defines the initial input variables and their types.
        2 Data dictionary contains:
            variable - name of the variable.
            type - type of the selected variable: 'numeric', 'factor'.

        Algorithm procedures:
        1 Imports the raw training data from a data frame or a .csv file.
        2 Extracts variables of the training data and adjusts variable types according to data dictionary.

    Examples:
        df, data_type_dict = import_training_data("./data/train.csv", data_type_dict, target_varname="Target")
    '''

    # default miss symbols
    miss_symbols = ["NA", "N/A", "null", "NULL", "?"] + [miss_symbols]

    # Error handlings
    if not isinstance(miss_symbols, list):
        print("The class of miss.symbols must be character.")
        exit()
    if encoding not in ["unknown", "UTF-8", "Latin-1"]:
        print("encoding can only be 'unknown', 'UTF-8' or 'Latin-1'.")
        exit()

    print("-----------------------")
    print("Training Data Importing")
    print("-----------------------")

    # read csv file
    data = pd.read_csv(data, na_values=miss_symbols, encoding=encoding)
    var_name_dict_data = pd.read_csv(open(dictionary), encoding=encoding)
    data_type_dict = dict([(i[1]['variable'], i[1]['type'])
                           for i in var_name_dict_data
                          .to_dict(orient='index')
                          .items()])

    # conduct data audit
    result_tag, audit_result_dict = explore_data_audit(data, explore_conf_dict, data_type_dict)
    if result_tag is False:
        return 'Audit Fail', audit_result_dict, ' '

    # Variable selection
    data = data[list(data_type_dict.keys())]

    # Extracts factor variables
    X_factor_data = pd.DataFrame()
    factor_vars = [i[0] for i in data_type_dict.items() if i[1] == 'factor']
    for col in [x for x in factor_vars if explore_conf_dict['target_varname'] not in x] :
        X_factor_data = pd.concat([X_factor_data, data[col].astype('category')], axis=1)

    # Extracts numeric variables
    numeric_vars = [i[0] for i in data_type_dict.items() if i[1] == 'numeric']
    X_numeric_data = pd.DataFrame(pd.DataFrame(data, columns=numeric_vars).apply(pd.to_numeric))

    X = pd.concat([X_factor_data, X_numeric_data], axis=1)
    Y = data[explore_conf_dict['target_varname']]
    df = pd.concat([X, Y], axis=1)

    print("The training data were imported.\n")
    # 打印日志
    # logger().info("The training data were imported.")
    return df, data_type_dict

