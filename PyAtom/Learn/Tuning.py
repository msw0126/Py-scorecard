# -*- coding: utf-8 -*- #
import gc
import os
import pandas as pd
from itertools import product
import numpy as np
from pandas import Series, DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from Learn.LearnPreprocessMain import *
from Act.ApplyPreprocess import *
from Learn.TrainModel import train_model
from  Act.ApplyModel import apply_model
from Utilities.PerformanceEvaluation import performance_evaluation

def model_validation(algorithm, X_train, Y_train, X_val, Y_val, hparms = dict(penalty = "l1", C = 0.1), metric = "ks", nbins_performance = 20):
    '''

    :param algorithm: name of the learning algorithm from the Algorithm Library.
                      1) 'glmnet' - lasso and Elastic-Net Regularized Generalized Linear Models
                      2) 'cart' - Classification and Regression Tree
    :param X_train: data frame of the input variables of the training data.
    :param Y_train: vector of the target variable of the training set.
    :param X_val: data frame of the input variables of the validation data.
    :param Y_val: vector of the target variable of the validation set.
    :param hparms: proportion of training data used as the internal training set in model validation for finding the optimal
    :param nbins_performance: list of hyperparameters of the given algorithm.
    :return: Returns the validation results.
             performance_table - data frame of performance table.
             auc - area under the ROC curve.
             ks - maximum ks of the performance table.
    '''

    # Model building
    model = train_model(X_train, Y_train, algorithm, hparms)

    # Model scoring
    probs = apply_model(model, X_val)

    # Performance evaluation
    obj = performance_evaluation(Y_val, probs, nbins_performance)

    return obj[metric]

@time_decrator('Tuning')
def tuning(data, data_type_dict, explore_conf_dict, metric = "ks", algorithm = "glmnet", prop_train = 0.7, nbins_performance = 20, output_dir = "Output"):
    '''
    Authors: Xianda Wu

    Description:
        Tune hyperparameters of a given algorithm by model validation.

    :param data: data frame of the input variables of the training data.
    :param data_type_dict: dict of the data dictionary
    :param explore_conf_dict:
    :param metric: major evaluation metric for hyperparameter tuning: 'ks', 'auc'.
    :param algorithm: name of the learning algorithm from the Algorithm Library.
                  1) 'glmnet' - lasso and Elastic-Net Regularized Generalized Linear Models
                  2) 'cart' - Classification and Regression Tree
    :param prop_train: proportion of training data used as the internal training set in model validation for finding the optimal algorithm and hyperparameters.
    :param nbins_performance: number of bins used in model validation for finding the optimal algorithm and hyperparameters.
    :param output_dir: name of the output directory.
    :return: hparms - optimal hyperparameter found by grid searching.
             results - optimal validation results.
             The following object will be exported into output.dir:
                "Tuning.csv" - records of validation performance of each algorithm with corresponding hyperparameters.
    Examples:
        from sklearn.datasets import make_hastie_10_2
        X, Y = make_hastie_10_2(random_state=0)
        X = X[:2000]
        Y = Y[:2000]
        Y[Y == -1] = 0
        obj = Tuning(X, Y, algorithm="glmnet")
    '''

    # Error handlings
    if not isinstance(prop_train, float):
        print("The class of prop_train must be numeric.")
        exit()
    if ((prop_train < 0.6) | (prop_train > 0.8)):
        print("prop_train should lie in [0.6,0.8]")
        exit()
    if not isinstance(nbins_performance, int):
        print("The class of nbins_performance must be numeric.")
        exit()
    if ((nbins_performance < 10) | (nbins_performance > 100)):
        print("nbins_performance should lie in [10,100] and nbins_performance must be smaller than the number of test records.")
        exit()
    if not isinstance(output_dir, str):
        print("The class of output_dir must be character.")
        exit()

    # Set up output directory
    folder = os.path.exists(output_dir)
    if not folder:
        os.makedirs(output_dir)

    target_varname = explore_conf_dict['target_varname']
    variable_name = [i for i in data.columns if i not in target_varname]
    X = data[variable_name]
    Y = data[target_varname]

    # Data Splitting
    ind0 = Y[Y == 0].index
    ind1 = Y[Y == 1].index
    ind_train = ind0[0:round(len(ind0) * prop_train)].append(ind1[0:round(len(ind1) * prop_train)])
    ind_val = list(set(Y.index).difference(set(ind_train)))

    X_train = X.iloc[ind_train,]
    Y_train = Y.iloc[ind_train]
    X_val = X.iloc[ind_val,].reset_index()
    Y_val = Y.iloc[ind_val].reset_index()[target_varname]

    data_train = pd.concat([X_train, Y_train], axis=1)
    data_val = pd.concat([X_val, Y_val], axis=1)
    del X, Y, X_train, Y_train
    gc.collect()

    # Data preprocessing over training data
    data_train, preprocess_conf_dict, preprocess_result = explore_preprocess_main(data_train, data_type_dict, explore_conf_dict)

    # Data preprocessing over test data
    X_val = apply_preprocess(X_val, preprocess_conf_dict)

    # Set default hyperparameters
    default_hparms = {'glmnet': dict(penalty=["l1", "l2"], C=[0.1, 1, 10, 100, 1000]),
                      'cart': dict(min_samples_split = [2, 5, 10], min_impurity_split = [1e-7, 1e-6, 1e-5])}
    hparms_pd = pd.DataFrame([row for row in product(*default_hparms[algorithm].values())],
                              columns=default_hparms[algorithm].keys())

    # hyperparameter tuning
    new_variable_name = preprocess_conf_dict['reserved_vars_one_hot_encoding']
    results = list()
    for i in range(0, hparms_pd.iloc[:, 0].size):
        hparms_i = hparms_pd.iloc[i]
        result_i = model_validation(algorithm, data_train[new_variable_name], data_train[target_varname],
                                    X_val, Y_val, hparms_i, metric, nbins_performance)
        results.append(result_i)

    # Calculate the optimal hyperparameter and optimal validation results
    ind_opt = results.index(max(results))
    hparms =  hparms_pd.iloc[ind_opt]
    results = results[ind_opt]

    return hparms, results