
from Utilities.DataProcess import *
from Utilities.LogTools import *


def cal_correlation(data_df):
    """
    calculate correlation coefficients of all variable pairs in the dataframe
    :param data_df: target dataframe
    :return: a dataframe of correlation coefficients
    """
    var_names = data_df.columns.tolist()
    cor_df = data_df.corr()
    cor_df = fill_missing(cor_df, dict([(var, 0) for var in var_names]))

    return cor_df


def find_collinearity_vars(cor_df, collinear_cutoff, weight_dict):
    """
    find variables pairs with high correlation, and get the varibales with lower weights to remove
    :param cor_df: correlation matrix
    :param collinear_cutoff: cut off value of correlation
    :param weight_dict: iv value of each variable
    :return: a list of variables that correlated with other variables and have lower weghts, 
     to remove
    """
    remove_vars = list()
    var_names = cor_df.columns.tolist()
    ncols = len(var_names)
    collinear_pairs_lst = list()

    # set diagonal value to 1 (self correlation)
    for i in range(ncols):
        cor_df[i, i] = 1
    # find pairs above collinear cutoff, true --> set 1; false -->set 0
    cut_df = cor_df.abs() >= collinear_cutoff
    # get each variables tatol correlated pairs, except itself (minus 1)
    col_sum_lst = (cut_df.sum(axis=0) - 1).tolist()

    # find correlated pairs' location
    cutoff_index_lst = [i for i in range(len(col_sum_lst)) if col_sum_lst[i] >= 1]
    # find correlated variables' name by their locations
    for i in cutoff_index_lst:
        lower_col_dict = cut_df.ix[(i + 1):, i].to_dict()
        for j in lower_col_dict.keys():
            if lower_col_dict[j] == 1:
                collinear_pairs_lst.append([var_names[i], j])
    del cut_df, cor_df
    gc.collect()

    # summarize collinear variables
    for pair in collinear_pairs_lst:
        tmp_pair = list(pair)
        if weight_dict[tmp_pair[0]] > weight_dict[tmp_pair[1]]:
            remove_vars.append(tmp_pair[1])
        else:
            remove_vars.append(tmp_pair[0])

    return list(set(remove_vars))


@time_decrator('collinearity analysis')
def collinearity_analysis_main(data_obj, explore_conf_dict):
    """
    conduct collinearity analysis
    :param data_obj: data object with target dataframe in it
    :param explore_conf_dict: explore configuration dict 
      with cutoff value of correlation coefficients in it
    :return: data object after collinear analysis and cutting off
    """
    collinear_cutoff = explore_conf_dict['collinearity_cutoff']
    variables = data_obj['info']['numeric_vars']
    if collinear_cutoff is not None and abs(collinear_cutoff) > 0:
        cor_df = cal_correlation(data_obj['df'][variables])
        weight_dict = data_obj['info']['iv_dict']
        remove_vars = find_collinearity_vars(cor_df, collinear_cutoff, weight_dict)
        data_obj['info']['remove_dict']['collinearity_vars'] = remove_vars
        data_obj['info']['numeric_vars'] = list(set(data_obj['info']['numeric_vars']) - set(remove_vars))
        data_obj['info']['factor_vars'] = list(set(data_obj['info']['factor_vars']) - set(remove_vars))
        data_obj['info']['remove_vars'] = list(set(data_obj['info']['remove_vars']) | set(remove_vars))
        data_obj['df'] = del_vars(data_obj['df'], remove_vars)

    return data_obj
