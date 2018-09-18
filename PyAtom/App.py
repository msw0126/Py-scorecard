import os
import pandas as pd
os.getcwd()
os.chdir("E:/DataBrain/PyAtom")
# pandas打印全部显示
pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 5000)
pd.set_option('display.width', 10000)

# Learn
from Learn.TrainRobot import train_robot

traindata = 'E:/DataBrain/Data/Fosun/train.csv'
dictionary = 'E:/DataBrain/Data/Fosun/dict.csv'
target_varname = 'GB.Indicator'
learn_dir = 'E:/DataBrain/Fosun_Learn'

train_robot(traindata, dictionary, target_varname, learn_dir)

# # Act
# from Act.ApplyRobot import apply_robot
#
# testdata = 'E:/DataBrain/Data/Fosun/test.csv'
# id_varname = "ID"
# act_dir = 'E:/DataBrain/Fosun_Act'
#
# apply_robot(learn_dir, testdata, id_varname, act_dir)
#
# # Test
# from Test.EvaluateRobot import evaluate_robot
#
# decision_data = "E:/DataBrain/Fosun_Act/DecisionData.csv"
# test_data = 'E:/DataBrain/Data/Fosun/test.csv'
# test_dir = 'E:/DataBrain/Fosun_Test'
#
# evaluate_robot(decision_data, test_data, target_varname, id_varname, test_dir)