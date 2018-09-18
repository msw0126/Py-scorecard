from Utilities.DataProcess import *
from Learn.DataTransform import one_hot_encoding
from Utilities.LogTools import *
import numpy as np


def discretize(data_df, numeric_breaks_dict):
    """
    discretize numeric variables
    :param data_df: a dataframe to discretize.
    :param numeric_breaks_dict: a map of numeric variables and their breaks info, dict
    :return: an discretized dataframe
    """
    discretize_dict = dict()
    for var in numeric_breaks_dict.keys():
        tmp_dict = numeric_breaks_dict[var]
        if '0' in tmp_dict['breaks_labels']:
            data_df.loc[data_df[var] < tmp_dict['breaks_lst'][1], var] = tmp_dict['breaks_lst'][1]
            data_df = fill_missing(data_df, {var: tmp_dict['breaks_lst'][0]})

        discretize_dict[var] = dict({'breaks_lst': tmp_dict['breaks_lst'],
                                     'breaks_labels': tmp_dict['breaks_labels']})

    data_df = discretize_numeric_vars(data_df, discretize_dict)

    return data_df


def factor_level_align(data_df, factor_level_dict, merge_dict, set_none=True):
    """
    align factor variables' level, set new level to None
    :param data_df: a dataframe to align
    :param factor_level_dict: a map of factor variables and their levels, dict.
    :param merge_dict:
    :param set_none:
    :return: an aligned dataframe
    """
    factor_vars = list(factor_level_dict.keys())
    for var in factor_vars:
        act_levels = list(data_df[var].unique())
        learn_levels = factor_level_dict[var]
        merge_levels = merge_dict[var]
        if len(merge_levels) > 1:
            learn_levels.remove('|'.join(merge_levels))
            learn_levels.extend(merge_levels)

        new_levels = list(set(act_levels) - set(learn_levels))
        if len(new_levels) > 0:
            for level in new_levels:
                data_df.loc[data_df[var] == level, var] = np.NaN

        data_df[var] = data_df[var].cat.add_categories(['null_flag'])
        data_df[var].fillna(value='null_flag')

        if len(merge_levels) > 1:
            data_df[var] = data_df[var].replace(to_replace=merge_levels,
                                                value='|'.join(merge_levels))

    return data_df


@time_decrator('data align')
def apply_preprocess(data_df, preprocess_conf_dict):
    """
    main function of data aligning
    :param data_df: a data object to be aligned
    :param preprocess_conf_dict: 
    :return: an aligned data object
    """
    data_df = discretize(data_df, preprocess_conf_dict['numeric_breaks_dict'])
    data_df = factor_level_align(data_df,
                                 preprocess_conf_dict['factor_level_dict'],
                                 preprocess_conf_dict['factor_merge_dict'])

    data_df, _ = one_hot_encoding(data_df, preprocess_conf_dict['reserved_vars'])
    for col in preprocess_conf_dict['reserved_vars_one_hot_encoding']:
        if col not in data_df.columns:
            data_df[col] = 0

    return data_df[preprocess_conf_dict['reserved_vars_one_hot_encoding']]
