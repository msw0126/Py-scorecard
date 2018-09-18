# -*- coding: utf-8 -*- #
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

def train_model(X, Y, algorithm="glmnet", hparms=dict(penalty = "l1", C = 0.1)):
    '''
    Predictive Model Training
    Authors: Xianda Wu

    Description:
      Builds a predictive model using a given combination of algorithm and hyperparameters over preprocessed training data.

    :param X: data frame of the input variables of the training data.
    :param Y: factor of the target variable of the training data. It must be a binary factor where the majority class is coded as 0 and the minority as 1.
    :param algorithm: name of the learning algorithm .
                  1) 'glmnet' - Lasso and Elastic-Net Regularized Generalized Linear Models
                  2) 'cart' - Classification and Regression Tree
    :param hparms: list of hyperparameters of the given algorithm.
    :return: Returns a model trained over the preprocessed training data.

    Examples:
      # Fosun data
      data.train = ImportTrainingData("../../../Data/Fosun/train.csv", "../../../Data/Fosun/dict.csv", "GB.Indicator")
      data.test = ImportTestData("../../../Data/Fosun/test.csv", data_train$dictionary, "GB.Indicator", "ID")
      obj_train = Preprocess(data_train["X"], data_train["Y"])
      obj_test = ApplyPreprocess(data_test{"X"], data_test{"Y"}, obj_train["preprocess_obj"])
      model = TrainModel(obj_train["X"], obj_train["Y"], algorithm = "glmnet", hparms = dict(penalty = "l1", C = 0.1))
      probs = ApplyModel(model, obj_test["X'])
      PerformanceEvaluation(obj_test["Y"], scores)
      ModelAnalysis(obj_train["X"], model)
    '''

    # Error handlings
    if algorithm == "glmnet":
        if not isinstance(hparms['penalty'], str):
            print("The class of penalty of hparms must be character.")
            exit()
        if hparms['penalty'] not in ["l1", "l2"]:
            print("penalty of hparms should only be 'l1', 'l2'.")
            exit()
        if not isinstance(hparms['C'], float):
            print("The class of C of hparms must be numeric.")
            exit()
        if hparms['C'] < 0:
            print("C of hparms should lager than 0.")
            exit()

    if algorithm == "cart":
        if not isinstance(hparms['min_samples_split'], float):
            print("The class of min_samples_split of hparms must be numeric.")
            exit()
        if hparms['min_samples_split'] <= 0:
            print("The min_samples_split of hparms must bigger than 0.")
            exit()
        if not isinstance(hparms['min_impurity_split'], float):
            print("The class of min_impurity_split of hparms must be numeric.")
            exit()
        if hparms['min_impurity_split'] <= 0 or hparms['min_impurity_split'] >= 1:
            print("min_impurity_split of hparms must lie in (0,1).")
            exit()

    # Train a model                                                                                                        # Lasso and Elastic-Net Regularized Generalized Linear Models
    if algorithm == "glmnet":
        model = LogisticRegression(penalty = hparms["penalty"], C = hparms["C"])
        model.fit(X, Y)

    # Classification  Tree
    if algorithm == "cart":
        model = DecisionTreeClassifier(min_samples_split = hparms["min_samples_split"], min_impurity_split = hparms["min_impurity_split"])
        model = model.fit(X, Y)
    print("Predictive model was built.\n \n")

    return model