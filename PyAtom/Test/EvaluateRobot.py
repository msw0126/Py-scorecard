
import time
import operator
import os
import numpy as np
from pandas import Series, DataFrame
import pandas as pd
from Utilities.PerformanceEvaluation import performance_evaluation


def evaluate_robot(decision_data, test_data, target_varname, id_varname, test_dir):
    '''
    Robot Evaluation
    Authors: Xianda Wu

    Description:
        Evaluates the performance of the predictive scores predicted by a trained Robot.

    Details:
        The requirements of decision data:
        id - ID of the test data for which to make predictions.
        scores - probabilities of the test records belonging to the positive class (class of Y=1) (from 0 to 1000).

        The requirements of test data:
          1 The first row specifies the names of all the variables.
        2 Each row corresponds to a record and each column corresponds to a variable.
        3 The types of the variables can only include 'numeric', 'factor'.
        4 If the target variable is included, then the target variable must be a binary factor with factor levels of (0,1) and no missing values are allowed in the target variable.
        5 The ID variable must be included in the test data.
        6 The test records cannot have duplicated IDs.
        7 The missing values are represented by "NA", "N/A", "null", "NULL", "?".
        8 The variable names cannot contain the symbol of "-".

        Algorithm procedures:
          1 Test Data Importing
        2 Decision Data Importing
        3 Decision Data and Test Data Merging
        4 Extracts true outcome
        5 Extracts scores
        6 Performance Evaluation
        7 Exports performance table


    :param decision_data: name of the .csv file of the decision data.
    :param test_data: name of the .csv file of the test data.
    :param target_varname: name of the target variable.
    :param id_varname: name of the ID variable.
    :param test_dir: name of the output directory of testing.
    :return: The following files will be exported into the output directory:
                "Test.txt" - report of Testing.
                "PerformanceTable.csv" - performance table.
    Examples:
        # German Credit data
        TrainRobot("../../../Data/GermanCredit/train.csv", "../../../Data/GermanCredit/dict.csv", "Target", "GermanCredit_Learn", algorithm = "rf")
        ApplyRobot("GermanCredit_Learn", "../../../Data/GermanCredit/test.csv", "id", "GermanCredit_Act")
        EvaluateRobot("GermanCredit_Act/DecisionData.csv", "../../../Data/GermanCredit/test.csv", "Target", "id", "GermanCredit_Test")
    '''

    # default miss symbols and encoding
    miss_symbols = ["NA", "N/A", "null", "NULL", "?"]
    encoding = "UTF-8"

    t0 = time.time()
    # Error handlings
    if not isinstance(decision_data, str):
        print("Invalid decision_data. decision_data - name of the .csv file of the decision data.")
        exit()
    if not isinstance(test_data, str):
        print("Invalid test_data. test_data - name of the .csv file of the test data.")
        exit()
    if not isinstance(target_varname, str):
        print("Invalid target_varname. target_varname - name of the target variable.")
        exit()
    if not isinstance(id_varname, str):
        print("Invalid id_varname. id_varname - name of the ID variable.")
        exit()
    if not isinstance(test_dir, str):
        print("Invalid test_dir. test_dir - name of the output directory of testing.")
        exit()

    # Set up output directory and file
    folder = os.path.exists(test_dir)
    if not folder:
        os.makedirs(test_dir)

    # Decision Data Importing
    decision_data = pd.read_csv(decision_data, na_values=miss_symbols, encoding=encoding)

    if decision_data.shape[0] == 0:
        print("The decision data specified by decision.data is invalid.")
        exit()
    if any(decision_data.isnull().any()):
        print("The ID and scores variable of the decision data cannot have missing values.")
        exit()

    # Test Data Importing
    test_data = pd.read_csv(test_data, na_values=miss_symbols, encoding=encoding, usecols=[id_varname, target_varname])
    if test_data.shape[0] == 0:
        print("The test data specified by test_data is invalid.")
        exit()
    if any(test_data.isnull().any()):
        print("The ID and target variable of the test data cannot have missing values.")
        exit()

    # Decision Data and Test Data Merging
    data = pd.merge(test_data, decision_data, on=id_varname)

    # Extracts true outcome
    Y = data[target_varname]
    if not (operator.eq(list(Y.unique()), [0, 1])
            or operator.eq(list(Y.unique()), [1, 0])):
        print("The target variable in test_data is invalid.")
        exit()

        # Extracts scores
    scores = data["scores"]

    # Performance Evaluation
    # Exports performance table
    performance = performance_evaluation(Y, scores)
    performance["performance_table"].to_csv(''.join([test_dir, '/', 'PerformanceTable.csv']), index=None)
