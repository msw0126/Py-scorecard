
from Test.EvaluateRobot import evaluate_robot

def test(decision_data, test_data, target_varname, id_varname, test_dir):
    evaluate_robot(decision_data, test_data, target_varname, id_varname, test_dir)