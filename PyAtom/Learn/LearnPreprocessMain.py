import copy
from Utilities.LogTools import *
from Learn.DataSummary import explore_data_statistics
from Learn.UnivariateAnalysis import explore_data_analysis
from Learn.DataClean import explore_data_clean
from Learn.DataTransform import *
from Learn.DataAudit import explore_data_audit
import datetime
import pandas as pd


def explore_config_4learn(explore_conf_dict, data_info_dict):
    """
    dump data process information 
    :param explore_conf_dict: configurations of exploring, dict.
    :param data_info_dict: a dict of data information 
    :return: None
    """
    preprocess_conf_dict = dict()

    preprocess_conf_dict['target_varname'] = explore_conf_dict.get('target_varname')
    preprocess_conf_dict['discretization'] = explore_conf_dict.get('discretization')
    preprocess_conf_dict['quantiles'] = explore_conf_dict.get('quantiles')

    preprocess_conf_dict['data_type_dict'] = data_info_dict.get('data_type_dict')
    preprocess_conf_dict['reserved_vars'] = data_info_dict.get('reserved_vars')
    preprocess_conf_dict['reserved_vars_one_hot_encoding'] = data_info_dict.get('reserved_vars_one_hot_encoding')
    preprocess_conf_dict['numeric_vars'] = data_info_dict.get('numeric_vars')
    preprocess_conf_dict['factor_vars'] = data_info_dict.get('factor_vars')
    preprocess_conf_dict['numeric_breaks_dict'] = data_info_dict.get('numeric_breaks_dict')
    preprocess_conf_dict['factor_merge_dict'] = data_info_dict.get('factor_merge_dict')

    preprocess_conf_dict['factor_level_dict'] = data_info_dict \
       .get('model_data_statistics_dict').get('factor_level_dict')

    return preprocess_conf_dict


def explore_preprocess_main(data_df, data_type_dict, explore_conf_dict):
    """
    main function of explore, to link explore process together, 
     and record consuming time of each step
    :param data_df: 
    :param data_type_dict:
    :param explore_conf_dict: configuration dictionary.
    :return: None
    """
    time_log_dict = dict()
    output_dict = dict()
    output_dict["Parameters used"] = copy.deepcopy(explore_conf_dict)
    start_time = datetime.datetime.now()

    # create data object
    data_df[explore_conf_dict['target_varname']] = pd.to_numeric(data_df[explore_conf_dict['target_varname']])
    data_obj = {'df': data_df, 'info': dict()}
    data_obj['info']['data_type_dict'] = data_type_dict
    data_obj['info']['numeric_vars'] = [i[0] for i in data_type_dict.items() if i[1] == 'numeric']
    data_obj['info']['factor_vars'] = [i[0] for i in data_type_dict.items() if i[1] == 'factor']
    data_obj['info']['factor_vars'].remove(explore_conf_dict['target_varname'])
    data_obj['info']['remove_vars'] = list()
    data_obj['info']['remove_dict'] = dict()

    # output config
    now_time, time_log_dict = time_recorder(start_time, 'parse configurations', time_log_dict)
    output_dict['time_recording'] = time_log_dict

    now_time, time_log_dict = time_recorder(now_time, 'data import and audit', time_log_dict)

    # original data summary
    ori_data_statistics_dict = explore_data_statistics(data_obj, explore_conf_dict, ori=True)
    data_obj['info']['ori_data_statistics_dict'] = ori_data_statistics_dict
    now_time, time_log_dict = time_recorder(now_time, 'original data summary', time_log_dict)
    output_dict['time_recording'] = time_log_dict

    # clean data
    data_obj = explore_data_clean(data_obj, explore_conf_dict)
    now_time, time_log_dict = time_recorder(now_time, 'clean data', time_log_dict)
    output_dict['time_recording'] = time_log_dict

    # univariate analysis
    data_obj = explore_data_analysis(data_obj, explore_conf_dict)
    now_time, time_log_dict = time_recorder(now_time, 'univariate analysis', time_log_dict)
    output_dict['time_recording'] = time_log_dict

    # data transform
    data_obj, sampled_method = explore_data_transform(data_obj, explore_conf_dict)
    now_time, time_log_dict = time_recorder(now_time, 'data transform', time_log_dict)
    output_dict['time_recording'] = time_log_dict

    # modeling data summary
    model_data_statistics_dict = explore_data_statistics(data_obj, explore_conf_dict, ori=False)
    data_obj['info']['model_data_statistics_dict'] = model_data_statistics_dict
    now_time, time_log_dict = time_recorder(now_time, 'modeling data summary', time_log_dict)
    output_dict['time_recording'] = time_log_dict

    # one hot encoding
    numeric_vars = data_obj['info']['numeric_vars']
    factor_vars = data_obj['info']['factor_vars']
    data_obj['df'], var_names_after_encoding = one_hot_encoding(data_obj['df'], numeric_vars + factor_vars)
    data_obj['info']['reserved_vars'] = numeric_vars + factor_vars
    data_obj['info']['reserved_vars_one_hot_encoding'] = var_names_after_encoding

    preprocess_conf_dict = explore_config_4learn(explore_conf_dict, data_obj['info'])

    now_time, time_log_dict = time_recorder(start_time, 'entire explore procedure', time_log_dict)
    output_dict['time_recording'] = time_log_dict

    return data_obj['df'], preprocess_conf_dict, data_obj['info']



