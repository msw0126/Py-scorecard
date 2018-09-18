# -*- coding: utf-8 -*- #
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from pandas import Series, DataFrame
import pandas as pd
import time

def apply_model(model, X, digits_probs = 6):
    '''
    Predictive Model Application
    Authors: Xianda Wu

    Description:
        Applies a trained model into the test data for predictive scorings.
        This function assumes the test data have already been preprocessed. If the test data are raw, then ApplyPreprocess.R needs to be invoked.

    :param model: model trained over the preprocessed training data.
    :param X: data frame of records for which to make predictions.
    :param digits_probs: number of digits of predictive probs. digits.probs should lie in [2,6].
    :return: Returns a vector of predictive probs ranging from 0 to 1.

    Details:
        Algorithm procedures:
        1 Predicts the probability of belonging to the class of Y=1 for each record of new data by using a trained model.

    Examples:
        # Fosun data
        data.train = ImportTrainingData("../../../Data/Fosun/train.csv", "../../../Data/Fosun/dict.csv", "GB.Indicator")
        data.test = ImportTestData("../../../Data/Fosun/test.csv", data_train$dictionary, "GB.Indicator", "ID")
        obj_train = Preprocess(data_train["X"], data_train["Y"])
        obj_test = ApplyPreprocess(data_test{"X"], data_test{"Y"}, obj_train["preprocess_obj"])
        model = TrainModel(obj_train["X"], obj_train["Y"], algorithm = "glmnet", hparms = dict(penalty = "l1", C = 0.1))
        probs = ApplyModel(model, obj_test["X'])
        PerformanceEvaluation(obj_test["Y"], scores, topNs.performance=c(100, 300))
        ModelAnalysis(obj_train["X"], model)
    '''

    t0 = time.time()

    print("----------------- \n")
    print("Model Application \n")
    print("----------------- \n")

    #  Error handlings
    if not isinstance(X, pd.DataFrame):
        print("X must be a data frame.")
        exit()
    if not isinstance(digits_probs, int):
        print("The class of digits_scores must be numeric.")
        exit()
    if ((digits_probs <2) | (digits_probs>6)):
        print("digits_scores should lie in [2,6].")
        exit()

    probs = pd.Series(model.predict_proba(X)[:,1])
    probs = round(probs, digits_probs)

    print("Predictive scores were returned.\n \n")
    print("Computing time of Model Application:\n")
    print(time.time() - t0)
    print("\n")
    return probs
