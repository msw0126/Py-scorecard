# -*- coding:utf-8 -*-
import datetime
import os
import pickle
import re

from Learn.ImportTrainingData import import_training_data
from Learn.LearnPreprocessMain import *
from Learn.ScoreCardGeneration import scorecard_bins_woe, get_score_card, card_to_json, scorecard_apply
from Learn.Tuning import *
from Learn.TrainModel import train_model
from Learn.ModelAnalysis import model_analysis
from Learn.UnivariateAnalysisPlot import *


def train_robot(data, dictionary, target_varname, learn_dir, metric = "ks", algorithm = "glmnet",
                quantiles = [0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1],
                sample_miss_cutoff = 0.95, variable_miss_cutoff = 0.95, variable_zero_cutoff = 0.95,
                max_nFactorLevels = 1024, nbins = 10, truncation_cutoff = 20, breaks_zero_cutoff = 0.3,
                iv_cutoff = 0.01, collinearity_cutoff = 0.9, unbalanced_cutoff = 5, onehot = True):
    '''

    :param data: name of the .csv file of the training data.
    :param dictionary: name of the .csv file of the data dictionary.
    :param target_varname: name of the target variable of the training data.
    :param learn_dir: name of the output directory of learning.
    :param metric: name of the major evaluation metric for algorithm and hyperparameter tuning.
                   'auc' - area under the ROC curve.
                   'ks' -  K-S statistic.
    :param algorithm: name of the learning algorithm from the Algorithm Library.
                     1) 'glmnet' - lasso and Elastic-Net Regularized Generalized Linear Models
                     2) 'cart' - Classification and Regression Tree
    :param quantiles: quantiles values.
    :param sample_miss_cutoff: cutoff value of proportion of missing values of each sample to be removed. sample_miss_cutoff must lie in [0.6,1].
    :param variable_miss_cutoff: cutoff value of proportion of missing values of each variable to be removed. variable_miss_cutoff must lie in [0.6,1].
    :param variable_zero_cutoff: cutoff value of proportion of zero values of each variable to be removed. variable_zero_cutoff must lie in [0.6,1].
    :param max_nFactorLevels: maximum number of factor levels of factor variables allowed. max_nFactorLevels must lie in [2,2000].
    :param nbins: number of bins for numeric variable. nbins must lie in [3,20].
    :param truncation_cutoff: coefficient of extreme value truncation for discretizing breaks settings of the numeric input variables. truncation_cutoff must lie in [2,20].
    :param breaks_zero_cutoff: cutoff value of proportion of zero values for seperate breaks settings. breaks_zero_cutoff must lie in [0.1,1]
    :param iv_cutoff: cutoff value of variable selection by IV filtering. See Predictive power of different values of IV in Details.
    :param collinearity_cutoff: cutoff value of the collinearity analysis for numeric variables. collinearity_cutoff must lie in [0,1].
                                If collinearity_cutoff==1, then no variables will be removed by collinearity analysis.
    :param unbalanced_cutoff: cutoff value of ratio of sample sizes of two classes for undersampling.
    :return:
    '''

    # set time
    time_log_dict = dict()
    output_dict = dict()
    start_time = datetime.datetime.now()

    # set parameter
    explore_conf_dict = dict()
    explore_conf_dict['target_varname'] = target_varname
    explore_conf_dict['metric'] = metric
    explore_conf_dict['algorithm'] = algorithm
    explore_conf_dict['quantiles'] = quantiles
    explore_conf_dict['sample_miss_cutoff'] = sample_miss_cutoff
    explore_conf_dict['variable_miss_cutoff'] = variable_miss_cutoff
    explore_conf_dict['variable_zero_cutoff'] = variable_zero_cutoff
    explore_conf_dict['max_nFactorLevels'] = max_nFactorLevels
    explore_conf_dict['nbins'] = nbins
    explore_conf_dict['truncation_cutoff'] = truncation_cutoff
    explore_conf_dict['breaks_zero_cutoff'] = breaks_zero_cutoff
    explore_conf_dict['iv_cutoff'] = iv_cutoff
    explore_conf_dict['collinearity_cutoff'] = collinearity_cutoff
    explore_conf_dict['unbalanced_cutoff'] = unbalanced_cutoff
    explore_conf_dict['onehot'] = onehot

    # Set up output directory and file
    folder = os.path.exists(learn_dir)
    if not folder:
        os.makedirs(learn_dir)

    # Training Data Importing
    df, data_type_dict = import_training_data(data, dictionary, explore_conf_dict)
    # Algorithm and Hyperparameter Tuning
    hparms, results = tuning(df, data_type_dict, explore_conf_dict, metric, algorithm, prop_train = 0.7, nbins_performance = 20, output_dir = ''.join([learn_dir, "/Tuning"]))

    # Data Preprocessing over Training Data(woe结果)
    df, preprocess_conf_dict, preprocess_result = explore_preprocess_main(df, data_type_dict,
                                                                            explore_conf_dict)
    # print(preprocess_result)
    # print([k for k, v in preprocess_result.items()])
    # # Univariate plot
    # plot_preprocess_result(preprocess_result, ''.join([learn_dir, "/Analysis"]))

    # Data Preprocessing over Training Data
    new_variable_name = preprocess_conf_dict['reserved_vars_one_hot_encoding']
    X = df[new_variable_name]
    # print(X.columns.values)
    # print(len(X.columns))
    # Y = df[target_varname]
    #
    # # Model Building and Model Analysis（model结果）
    # model = train_model(X, Y, algorithm, hparms)
    # weights_df = model_analysis(X, model, algorithm, ''.join([learn_dir, '/ModelAnalysis']), 2)
    #
    # # Robot Creation
    # Robot = dict()
    # Robot['model'] = model
    # Robot['preprocess_conf_dict'] = preprocess_conf_dict
    # Robot['data_type_dict'] = data_type_dict
    #
    # pickle.dump(Robot, open(''.join([learn_dir, '/Robot.pickle']), 'wb'))
    # print("A trained Robot was saved.")
    #
    # # Learn time
    # now_time, time_log_dict = time_recorder(start_time, 'Total computing time of learning', time_log_dict)
    # print(time_log_dict)
    #
    """
        评分卡生成
    """
    Robot = pickle.load(open(''.join([learn_dir, '/Robot.pickle']), 'rb'))
    bins_woe_df, factor_vars_variable_and_lable = scorecard_bins_woe(preprocess_result)
    woe_var_list = bins_woe_df['variable'].values
    train_col_list = X.columns.values
    remove_vars = list(set(woe_var_list).difference(set(train_col_list)))  # 差集
    bins_woe_df = bins_woe_df[~bins_woe_df['variable'].isin(remove_vars)].reset_index(drop=True)
    # # print(bins_woe_df)
    # # print("len(bins_woe_df):{}".format(len(bins_woe_df['variable'].values)))
    X_train_columns = X.columns
    model = Robot['model']
    score_card_df = get_score_card(bins_woe_df, model, X_train_columns, factor_vars_variable_and_lable)
    print(score_card_df)
    score_card_json = card_to_json(score_card_df)
    # print("评分卡json格式：{}".format(card_to_json(score_card)))
    # train_data = pd.read_csv("E:/DataBrain/Data/Fosun/train.csv")
    # train_data = train_data.drop('ID', axis=1).drop('GB.Indicator', axis=1)
    # score = scorecard_apply(train_data, score_card_df, preprocess_result)
    # # # print(score)
