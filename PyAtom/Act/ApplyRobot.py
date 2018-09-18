import datetime
import os
import pickle
import pandas as pd
from Act.ImportTestData import import_test_data
from Act.ApplyPreprocess import *
from Act.ApplyModel import apply_model
from Utilities.PerformanceEvaluation import performance_evaluation


def apply_robot(learn_dir, data, id_varname, act_dir, target_varname=None):
    '''

    :param learn_dir: name of the output directory of learning. The trained Robot "Robot.RData" must be included in learn.dir.
    :param data: name of the .csv file of the test data.
    :param id_varname: name of the ID variable of the test data.
    :param act_dir: name of the output directory of acting.
    :param target_varname: name of the target variable of the test data. If NULL, then the target variable is not included in the test data.
    :return:
    '''

    # set time
    time_log_dict = dict()
    start_time = datetime.datetime.now()

    # Set up output directory and file
    folder = os.path.exists(act_dir)
    if not folder:
        os.makedirs(act_dir)

    # Robot Importing
    Robot = pickle.load(open(''.join([learn_dir, '/Robot.pickle']), 'rb'))
    model = Robot['model']
    preprocess_conf_dict = Robot['preprocess_conf_dict']
    data_type_dict = Robot['data_type_dict']

    # Test Data Importing
    if target_varname is None:
        target_varname = preprocess_conf_dict['target_varname']
    df, id = import_test_data(data, data_type_dict, id_varname, target_varname)

    # Data Preprocessing over Test Data
    X = apply_preprocess(df, preprocess_conf_dict)

    # Predictive Scoring
    probs = apply_model(model, X, digits_probs=6)

    # Decision Data Generation
    decision_data = pd.concat([id, probs], axis=1)
    decision_data.columns = [id_varname, "scores"]
    decision_data.to_csv(''.join([act_dir, '/', 'DecisionData.csv']), index=None)

    # Performance Evaluation (if the target variable of the test data exists)
    if target_varname in list(df.columns):
        Y = df[target_varname]
        performance = performance_evaluation(Y, probs)
        performance['performance_table'].to_csv(''.join([act_dir, '/','PerformanceTable.csv']), index=None)

    # Act time
    now_time, time_log_dict = time_recorder(start_time, 'Total computing time of Act', time_log_dict)
    print(time_log_dict)


