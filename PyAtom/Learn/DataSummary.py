from Utilities.DataProcess import *
from Utilities.LogTools import *


def factor_vars_levels(data_df):
    """
    calculate each factor variable's unique levels, amount of each level, the most frequent level
    :param data_df: target dataframe with only factor variables
    :return: three dicts, 
    a map of factor vars and their levels with each level amount (var1:{level1:amount, ...}), 
    a map of vars and their unique levels list (var1:[level1, level2,...]),
    a map of factor vars and their most frequent level (var1:level ...),
    """
    factor_level_am_dict = dict()
    factor_level_dict = dict()
    factor_mfreq_dict = dict()
    for var in data_df.columns:
        level_amount_dict = cal_vars_levels_amount(data_df, var)
        factor_level_am_dict[var] = level_amount_dict
        factor_level_dict[var] = list(level_amount_dict.keys())
        factor_mfreq_dict[var] = sorted(level_amount_dict.items(),
                                        key=lambda d: d[1], reverse=True)[0][0]

    return factor_level_am_dict, factor_level_dict, factor_mfreq_dict


@time_decrator('data summary')
def explore_data_statistics(data_obj, explore_conf_dict, ori=True):
    """
    main function of data summary  
    :param data_obj: a data object with target dataframe in int 
    :param explore_conf_dict: configurations of exploring, dict.
    :param ori: 
    :return: data summary result, dict.
    """
    data_statistics_dict = dict()
    numeric_vars = data_obj['info']['numeric_vars']
    factor_vars = data_obj['info']['factor_vars']
    tmp_df = data_obj['df']

    nrows, ncols = data_obj['df'].shape
    data_statistics_dict['nrows'] = nrows
    data_statistics_dict['ncols'] = ncols

    if ori:
        quantiles = explore_conf_dict['quantiles']
        data_statistics_dict['quantiles'] = quantiles
        data_statistics_dict['numeric_quantiles_dict'] = cal_vars_quantiles(tmp_df, numeric_vars, quantiles)
        data_statistics_dict['numeric_median_dict'] = cal_vars_median(tmp_df, numeric_vars)
    else:
        data_statistics_dict['numeric_bin_am_dict'], \
        data_statistics_dict['numeric_bin_dict'], \
        data_statistics_dict['numeric_bin_mfreq_dict'] = factor_vars_levels(tmp_df[numeric_vars])

    data_statistics_dict['factor_level_am_dict'], \
    data_statistics_dict['factor_level_dict'], \
    data_statistics_dict['factor_mfreq_dict'] = factor_vars_levels(tmp_df[factor_vars])

    data_statistics_dict['target_distribution']\
        = cal_vars_levels_amount(tmp_df, explore_conf_dict['target_varname'])

    data_statistics_dict['var_miss_dict'] = cal_vars_miss(tmp_df[numeric_vars + factor_vars])

    return data_statistics_dict



