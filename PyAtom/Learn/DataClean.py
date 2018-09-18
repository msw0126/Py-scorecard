from Utilities.DataProcess import *
from Utilities.LogTools import *


def find_const_vars(statistics_dict, numeric_vars, factor_vars, nrows):
    """
    find constant variables to remove
    :param statistics_dict: statistic information of variables, include:
     1.missing amount; 2. levels of factor variables; 3. std of numeric variables.
    :param numeric_vars: numeric variables
    :param factor_vars: factor variables
    :param nrows: number of rows
    :return: a list of constant variable names to remove, list of str.
    """
    remove_lst = list()
    for var in numeric_vars:
        if statistics_dict[var]['count'] == 0 \
                or (statistics_dict[var]['count'] == nrows
                    and statistics_dict[var]['std'] == 0):
            remove_lst.append(var)
    for var in factor_vars:
        if statistics_dict[var]['count'] == 0 \
                or (statistics_dict[var]['count'] == nrows
                    and statistics_dict[var]['unique']) == 1:
            remove_lst.append(var)

    return remove_lst


def find_high_miss_ratio_samples(data_df, sample_miss_cutoff):
    """
    find samples with high ratio of missing elements
    :param data_df: target dataframe
    :param sample_miss_cutoff: cutoff value of missing ratio
    :return: a list of sample indexes to remove, list of int 
    """

    remove_lst = list()
    ncols = data_df.shape[1]
    row_miss_lst = cal_samples_miss(data_df)
    row_miss_ratio_lst = [i * 1.0 / ncols for i in row_miss_lst]
    for i in range(len(row_miss_ratio_lst)):
        if row_miss_ratio_lst[i] >= sample_miss_cutoff:
            remove_lst.append(i)

    return remove_lst


def find_high_miss_ratio_vars(nrows, statistics_dict, variable_miss_cutoff):
    """
    find variables with high ratio of missing elements
    :param nrows: number of rows
    :param statistics_dict: statistic information of variables, include:
     1.missing amount of variables.
    :param variable_miss_cutoff: cutoff value of missing ratio
    :return: a list of variables indexes to remove, list of str 
    """
    remove_lst = list()
    for var in statistics_dict.keys():
        if (nrows - statistics_dict[var]['count']) * 1.0 / nrows >= variable_miss_cutoff:
            remove_lst.append(var)

    return remove_lst


def find_high_zero_ratio_vars(data_df, numeric_vars, variable_zero_cutoff):
    """
    find variables with high ratio of zeros
    :param data_df: target data frame
    :param numeric_vars: numeric variables
    :param variable_zero_cutoff: cutoff value of zero ratio
    :return: a list of variables to remove, list of str 
    """
    zero_ratio_dict = cal_zero_ratio(data_df, numeric_vars)
    remove_lst = list()
    for var in numeric_vars:
        if zero_ratio_dict[var] >= variable_zero_cutoff:
            remove_lst.append(var)

    return remove_lst


def find_large_levels_vars(statistics_dict, factor_vars, variable_levels_cutoff):
    """
    find variables with large levels to remove
    :param statistics_dict: statistic information of variables, include:
     1. levels of factor variables.
    :param factor_vars: factor variables
    :param variable_levels_cutoff: cutoff value of max levels for each factor variable
    :return: a list of variables to remove, list of str 
    """
    remove_lst = list()
    for var in factor_vars:
        if statistics_dict[var]['unique'] >= variable_levels_cutoff:
            remove_lst.append(var)

    return remove_lst


@time_decrator('clean data')
def explore_data_clean(data_obj, explore_conf_dict):
    """
    main function of data cleaning
    :param data_obj: a data object with a dataframe in int
    :param explore_conf_dict: a dict of configurations for exploring
    :return: an object with cleaned dataframe in int
    """
    remove_dict = dict()
    variables = data_obj['info']['numeric_vars'] + data_obj['info']['factor_vars']

    # row wise
    # find samples' missing ratio
    sample_miss_cutoff = explore_conf_dict['sample_miss_cutoff']
    remove_dict['miss_samples'] = find_high_miss_ratio_samples(data_obj['df'][variables], sample_miss_cutoff)
    data_obj['df'] = del_samples(data_obj['df'], remove_dict['miss_samples'])

    # column wise
    nrows, ncols = data_obj['df'].shape
    tmp_df = data_obj['df'][variables]
    numeric_vars = data_obj['info']['numeric_vars']
    factor_vars = data_obj['info']['factor_vars']
    # get statistical information of all variables
    statistics_dict = get_describe_result(tmp_df, numeric_vars, factor_vars)
    # find const vars
    remove_dict['constant_vars'] = find_const_vars(statistics_dict, numeric_vars, factor_vars, nrows)
    # find vars with high missing ratio
    variable_miss_cutoff = explore_conf_dict['variable_miss_cutoff']
    remove_dict['miss_vars'] = find_high_miss_ratio_vars(nrows, statistics_dict, variable_miss_cutoff)
    # find numeric vars with high zero ratio
    variable_zero_cutoff = explore_conf_dict['variable_zero_cutoff']
    remove_dict['zero_vars'] = find_high_zero_ratio_vars(tmp_df, numeric_vars, variable_zero_cutoff)
    # find factor vars with large levels
    variable_levels_cutoff = explore_conf_dict['max_nFactorLevels']
    remove_dict['levels_vars'] = find_large_levels_vars(statistics_dict, factor_vars,
                                                        variable_levels_cutoff)
    # removing
    data_obj['info']['remove_dict'].update(remove_dict)
    remove_vars = list(set(remove_dict['constant_vars'] + remove_dict['miss_vars']
                           + remove_dict['zero_vars'] + remove_dict['levels_vars']))

    data_obj['info']['numeric_vars'] = list(set(data_obj['info']['numeric_vars']) - set(remove_vars))
    data_obj['info']['factor_vars'] = list(set(data_obj['info']['factor_vars']) - set(remove_vars))
    data_obj['info']['remove_vars'] = list(set(data_obj['info']['remove_vars']) | set(remove_vars))

    data_obj['df'] = del_vars(data_obj['df'], remove_vars)
    print(data_obj['df'].shape)

    return data_obj


