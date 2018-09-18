# -*- coding: utf-8 -*- #
from sklearn import metrics
import operator
import numpy as np
from pandas import Series, DataFrame
import pandas as pd

def performance_evaluation(Y, scores, nbins_performance = 20, digits_performance = 2):
    '''
    Evaluation of Predictive Performance
    Authors: Xianda Wu

    Description:
        Evaluates the predictive performance by comparing the predictive scorings to the true outcomes.

    :param Y: factor of the target variable of the test data. It must be a binary factor where the factor levels are (0,1).
    :param scores: vector of predictive scorings ranging from 0 to 1000.
    :param nbins_performance: number of bins of the performance table. nbins_performance should lie in [10,100] and nbins_performance must be smaller than the number of test records.
    :param digits_performance: number of digits of performance evaluations. digits_performance should lie in [0,4].
    :return: performance_table - data frame of performance table.
             auc - area under the ROC curve.
             ks - maximum ks of the performance table.
    Details:
        Algorithm procedures:
        1 Audits the target variable of the test data and the predictive scores.
        2 Ranks the test records by the predictive scores and splits the test records into nbins_performance bins.
        3 Calculates AUC (area under the ROC curve).
        4 Generates the performance table with the following measures of each bin:
        CumProp - cumulative proportion of test record.
        Size - number of test records.
        MinScore - minimum value of scores.
        MeanScore - mean value of scores.
        MaxScore - maximum value of scores.
        PositiveSize - number of positive records (Y = 1).
        NegativeSize - number of negative records (Y = 0).
        Precision - cumulative sum of positive records / cumulative sum number of records.
        Recall - cumulative sum of positive records / total number of positive records.
        ks - Recall - cumulative sum of negative records / total number of negative records.
        5 Calculates KS by finding the maximum value of ks of performance table.

    Examples:
      # Fosun data
      data.train = ImportTrainingData("../../../Data/Fosun/train.csv", "../../../Data/Fosun/dict.csv", "GB.Indicator")
      data.test = ImportTestData("../../../Data/Fosun/test.csv", data_train$dictionary, "GB.Indicator", "ID")
      obj_train = Preprocess(data_train["X"], data_train["Y"])
      obj_test = ApplyPreprocess(data_test{"X"], data_test{"Y"}, obj_train["preprocess_obj"])
      model = TrainModel(obj_train["X"], obj_train["Y"], algorithm = "glmnet", hparms = dict(penalty = "l1", C = 0.1))
      probs = ApplyModel(model, obj_test["X'])
      PerformanceEvaluation(obj_test['Y'], scores, topNs.performance=c(100, 300))
    '''

    print("---------------------- \n")
    print("Performance Evaluation \n")
    print("---------------------- \n")

    Y = pd.Series(Y)
    scores = pd.Series(scores)

    # Audits the target variable and the predictive scores
    if Y.isnull().sum() > 0:
        print("The target variable cannot have missing values.")
        exit()
    if not (operator.eq(list(pd.unique(Y)), [0, 1])
            or operator.eq(list(pd.unique(Y)), [1, 0])):
        print("The target variable in Performance Evaluation is invalid.")
        exit()
    if scores.isnull().sum() > 0:
        print("The target variable cannot have missing values.")
        exit()
    if (len(Y) != len(scores)):
        print("Y and scores are not compatible.")
        exit()
    if not isinstance(nbins_performance, int):
        print("The class of nbins_performance must be numeric.")
        exit()
    if ((nbins_performance < 10) | (nbins_performance > 100)):
        print("nbins_performance should lie in [10,100] and nbins_performance must be smaller than the number of test records.")
        exit()
    if not isinstance(digits_performance, int):
        print("The class of digits_performance must be numeric.")
        exit()
    if ((digits_performance < 0) | (digits_performance > 4)):
        print("digits_performance should lie in [0,4].")
        exit()

    # Calculates AUC
    auc = metrics.roc_auc_score(Y, scores)

    # Basics
    i_appended = None
    n = len(Y.index)
    n1 = Y[Y == 1].count()
    n0 = n - n1

    # Ranks the test records by scores
    ind = pd.Series(pd.qcut(-scores, nbins_performance, labels=False))

    # Initialization
    MinScore = list()
    MeanScore = list()
    MaxScore = list()
    Size = list()
    Precision = list()
    PositiveSize = list()
    NegativeSize = list()

    # Calculates variables of performance table
    CumProp = pd.Series(range(round(100 / nbins_performance), 101, round(100/nbins_performance)))
    for i in list(set(ind)):
        j = ind[ind == i].index
        Yj = Y[j]
        Pj = scores[j]
        Size.append(len(j))
        MinScore.append(min(Pj))
        MeanScore.append(np.mean(Pj))
        MaxScore.append(max(Pj))
        PositiveSize.append(Yj[Yj == 1].count())
        NegativeSize.append(Yj[Yj == 0].count())

    # Generates performance table
    performance_table = pd.DataFrame({'CumProp':CumProp, 'Size':Size, 'MinScore':MinScore, 'MeanScore':MeanScore,
                                     'MaxScore':MaxScore, 'PositiveSize':PositiveSize, 'NegativeSize':NegativeSize})
    performance_table['Precision'] = performance_table['PositiveSize'].cumsum() / performance_table['Size'].cumsum() * 100
    performance_table['Recall'] = performance_table['PositiveSize'].cumsum() / n1 * 100
    performance_table['ClassY1Cap'] = performance_table['PositiveSize'].cumsum() / n1 * 100
    performance_table['ClassY0Cap'] = performance_table['NegativeSize'].cumsum() / n0 * 100
    performance_table['ks'] = performance_table['ClassY1Cap'] - performance_table['ClassY0Cap']

    # Rounding
    performance_table['MeanScore'] = round(performance_table['MeanScore'], digits_performance)
    performance_table['Precision'] = round(performance_table['Precision'], digits_performance)
    performance_table['Recall'] = round(performance_table['Recall'], digits_performance)
    performance_table['ks'] = round(performance_table['ks'], digits_performance)
    obj = {'performance_table':performance_table, 'auc':auc, 'ks':max(performance_table['ks'])}
    return obj

