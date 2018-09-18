# -*- coding: utf-8 -*- #
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from pandas import Series, DataFrame
import pandas as pd
import os

def model_analysis(X, model, algorithm = "glmnet", output_dir = "ModelAnalysis", digits_weights = 2):
    '''
    Predictive Model Analysis
    Authors: Xianda Wu

    Description:
        S

    :param X: data frame of the input variables of the training data.
    :param model: trained predictive model.
    :param algorithm: name of the learning algorithm .
                  1) 'glmnet' - Lasso and Elastic-Net Regularized Generalized Linear Models
                  2) 'cart' - Classification and Regression Tree
    :param output_dir: name of the output directory.
    :param digits_weights: number of digits of variable weights returned by a trained model. digits_weights should lie in [0,4].
    :return: Returns the variable weights from a trained model.
             The following files will be exported into the output directory:
                "VariableWeights.csv" - file of variable weights returned by a trained model.
                "VariableWeightsPlot.pdf" - plot of distribution of variable weights returned by a trained model.
                "DecisionRules.txt" - decision rules found by 'cart'.

    Details:
        Algorithm procedures:
        1 Evaluates variable weights from a trained model according to the learning algorithm used.
        2 Returns decision rules from a trained model for the CART algorithm.
        3 Exports the plot of distribution of variable weights.
        4 Exports the .csv file of variable weights.

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

    print("-------------- \n")
    print("Model Analysis \n")
    print("-------------- \n")

    # Error handlings
    if not isinstance(X, pd.DataFrame):
        print("X must be a data frame.")
        exit()
    if not isinstance(output_dir, str):
        print("The class of output_dir must be character.")
        exit()
    if not isinstance(digits_weights, int):
        print("The class of digits_weights must be numeric.")
        exit()
    if ((digits_weights<0) | (digits_weights>4)):
        print("digits_weights should lie in [0,4].")
        exit()

    # model analysis
    if algorithm == "glmnet":
        coef = model.coef_
        weights_df = pd.DataFrame({'variable': list(X.columns), 'weights':coef[0,].tolist()})
    if algorithm == "cart":
        weights_df = pd.DataFrame({'variable': list(X.columns), 'weights':model.feature_importances_})

    # Set up output directory
    folder = os.path.exists(output_dir)
    if not folder:
        os.makedirs(output_dir)

    # Exports the .csv file of variable weights
    weights_df = weights_df.sort_index(by = 'weights', ascending = False)
    weights_df['weights'] = round(weights_df['weights'], digits_weights)
    weights_df.to_csv(''.join([output_dir, '/', 'VariableWeights.csv']), index=None)
    return weights_df
