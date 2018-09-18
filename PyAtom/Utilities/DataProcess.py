# -*- coding:utf-8 -*-

import pandas as pd
import gc


def cal_vars_median(data_df, numeric_vars):
    """
    calculate medians of numeric variables
    :param data_df: target dataframe
    :param numeric_vars: numeric variables, list of str
    :return: a map of numeric variables and their median values, dict.
    """
    median_dict = data_df[numeric_vars].median().to_dict()

    return median_dict


def cal_vars_levels_amount(data_df, factor_var):
    """
    calculate amount of each level of a factor variable
    :param data_df: target dataframe
    :param factor_var: target factor variable, str.
    :return: a map of levels and their amounts, dict.
    """
    level_amount_dict = data_df.groupby(by=factor_var).count().iloc[:, 0].to_dict()

    return level_amount_dict


def cal_samples_miss(data_df):
    """
    calculate missing value amounts of each sample
    :param data_df: target dataframe
    :return: a list of missing value amounts of each sample
    """
    row_miss_lst = data_df.isnull().sum(axis=1).tolist()

    return row_miss_lst


def cal_zero_ratio(data_df, numeric_vars):
    """
    calculate zeros ratio of numeric variables
    :param data_df: target dataframe
    :param numeric_vars: numeric variables, list of str
    :return: a map of variables and their zeros ratio, dict.
    """
    nrows = data_df.shape[0]
    zeros_dict = dict()
    for var in numeric_vars:
        nzero = data_df[data_df[var] == 0].shape[0]
        zeros_dict[var] = (1.0 * nzero / nrows)

    return zeros_dict


def del_samples(data_df, remove_index_lst=None):
    """
    delete samples by sample index
    :param data_df: target dataframe
    :param remove_index_lst: a list of row index that to be remove from the dataframe 
    :return: a dataframe after removing pointed rows
    """
    if len(remove_index_lst) > 0:
        data_df = data_df.drop(remove_index_lst, axis=0)

        gc.collect()

    return data_df


def cal_vars_miss(data_df):
    """
    calculate amount of missing values of each variables
    :param data_df: target dataframe
    :return: a map of variables and their missing ratio, dict.
    """
    var_miss_dict = data_df.isnull().sum(axis=0).to_dict()

    return var_miss_dict


def cal_vars_std(data_df, numeric_vars):
    """
    calculate standard deviation of each variable
    :param data_df: target dataframe
    :param numeric_vars: numeric variables, list of str
    :return: a map of variables and their standard deviation, dict.
    """
    col_std_dict = data_df[numeric_vars].std().to_dict()

    return col_std_dict


def cal_vars_levels(data_df, factor_vars):
    """
    collect different levels of each factor variable
    :param data_df: a dataframe with only factor variables
    :param factor_vars: factor variables, a list of str.
    :return: a map of variables and their different levels as a list, dict. 
    """
    levels_dict = dict(zip(factor_vars, [list(data_df[i].unique()) for i in factor_vars]))

    return levels_dict


def del_vars(data_df, remove_var_lst):
    """
    delete variables
    :param data_df: target dataframe
    :param remove_var_lst: a list of variable to be removed from the dataframe
    :return: a dataframe after removing pointed variables
    """
    if len(remove_var_lst) > 0:
        data_df = data_df.drop(remove_var_lst[:], axis=1)

        gc.collect()

    return data_df


def random_sampling(data_df, ratio=None, sub_num=None):
    """
    random sampling by ratio or sample number
    :param data_df: target dataframe
    :param ratio: sampling ratio, float
    :param sub_num: sampling number, int
    :return: a sampled dataframe
    """
    if ratio is None and sub_num is not None:
        ratio = 1.0*sub_num / data_df.shape[0]
    assert isinstance(ratio, float) and 0 < ratio < 1, \
        'sample ratio or number is out of range !'

    sampled_df = data_df.sample(frac=ratio, random_state=7).sort_index()

    return sampled_df


def fill_missing(data_df, fill_dict):
    """
    fill missing value
    :param data_df: target dataframe
    :param fill_dict: a dict that give a value for each variable to fill
    :return: a filled dataframe
    """
    if isinstance(fill_dict, dict) and len(fill_dict.keys()) > 0:
        data_df = data_df.fillna(value=fill_dict)

    return data_df


def cal_quantiles(data_df, numeric_var, percentiles=None, omit_zero=False, rand_num=None):
    """
    calculate quantiles' values for a specific numeric varibale
    :param data_df: target dataframe
    :param numeric_var: name of numeric variable to calculate quantiles' values, str
    :param percentiles: percentiles to calculate their corresponding quantiles, list of float
    :param omit_zero: whether omit zeros, bool
    :param rand_num: random sampling number if want to calculate quantiles on sampling data set
    :return: a list of quantiles' values
    """
    if percentiles is None:
        percentiles = [0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]

    if rand_num:
        tmp_df = random_sampling(data_df, sub_num=rand_num)
    else:
        tmp_df = data_df

    if omit_zero:
        quantiles_df = tmp_df[tmp_df[numeric_var] != 0][numeric_var]\
            .quantile(q=percentiles)
    else:
        quantiles_df = tmp_df[numeric_var] \
            .quantile(q=percentiles)

    del tmp_df
    gc.collect()

    quantile_lst = quantiles_df.tolist()
    for i in range(len(quantile_lst)):
        if quantile_lst[i] is not None:
            quantile_lst[i] = float(quantile_lst[i])
        else:
            quantile_lst[i] = None
    if None in quantile_lst:
        quantile_lst = list()

    del quantiles_df
    gc.collect()

    return quantile_lst


def cal_vars_quantiles(data_df, numeric_vars, percentiles=None):
    """
    calculate quantiles' values for a specific numeric varibale
    :param data_df: target dataframe
    :param numeric_vars: names of numeric variables to calculate quantiles' values, str
    :param percentiles: percentiles to calculate their corresponding quantiles, list of float
    :return: a map of numeric vars and their list of quantiles' values, dict
    """

    if percentiles is None:
        percentiles = [0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]

    quantiles_df = data_df[numeric_vars] \
        .quantile(q=percentiles)

    quantiles_dict = quantiles_df.to_dict()
    for i in numeric_vars:
        tmp_quantiles_lst = []
        sub_quantiles_dict = quantiles_dict[i]
        for j in percentiles:
            if sub_quantiles_dict[j] is not None:
                tmp_quantiles_lst.append(float(sub_quantiles_dict[j]))
            else:
                tmp_quantiles_lst.append(None)
        quantiles_dict[i] = tmp_quantiles_lst

    del quantiles_df
    gc.collect()

    return quantiles_dict


def discretize_numeric_vars(data_df, discretize_dict):
    """
    discretize numeric variables
    :param data_df: target dataframe
    :param discretize_dict: a map of numeric variables and their cut informations
    :return: a discretized dataframe
    """
    for var in discretize_dict.keys():
        tmp_dict = discretize_dict[var]
        breaks_lst = tmp_dict['breaks_lst']
        max_cut = max(breaks_lst)
        data_df.loc[data_df[var] > max_cut, var] = max_cut
        min_cut = min(breaks_lst)
        data_df.loc[data_df[var] < min_cut, var] = min_cut

        tmp_breaks = breaks_lst[:]
        tmp_breaks[0] = tmp_breaks[0] - 1
        tmp_breaks[-1] = tmp_breaks[-1] + 1

        data_df[var] = pd.cut(data_df[var], bins=tmp_breaks, right=False, labels=tmp_dict['breaks_labels'],
                              precision=5, include_lowest=True)

        data_df[var] = pd.to_numeric(data_df[var])

    return data_df


def get_describe_result(data_df, numeric_vars=None, factor_vars=None):
    """
    get summary result from describe funtion of h2o by calling the function once
    calculated content include by order: type, mins, mean, maxs, sigma, zero_count, missing_count
    :param data_df: target dataframe
    :param numeric_vars: numeric variables, list of str
    :param factor_vars: factor variables, list of str
    :return: a map of numeirc variables and their maps of statistics and their values
    """
    statistics_dict = dict()
    variables = list()
    if isinstance(numeric_vars, list):
        variables.extend(numeric_vars)
    if isinstance(factor_vars, list):
        variables.extend(factor_vars)

    summary_dict = data_df[variables].describe(include='all').to_dict()

    if isinstance(numeric_vars, list):
        for var in numeric_vars:
            tmp_dict = dict()

            tmp_dict['max'] = summary_dict[var]['max']
            tmp_dict['min'] = summary_dict[var]['min']

            tmp_dict['mean'] = summary_dict[var]['mean']
            tmp_dict['std'] = summary_dict[var]['std']
            tmp_dict['count'] = summary_dict[var]['count']

            statistics_dict[var] = tmp_dict

    if isinstance(factor_vars, list):
        for var in factor_vars:
            tmp_dict = dict()

            tmp_dict['count'] = summary_dict[var]['count']
            tmp_dict['unique'] = summary_dict[var]['unique']

            statistics_dict[var] = tmp_dict
    result_dict = statistics_dict.copy()
    del statistics_dict
    gc.collect()

    return result_dict

def check_parameter(explore_conf_dict):
    '''
    check import parameters
    :param explore_conf_dict:
    :return: audit result, dict
    '''

    audit_result_dict = dict()
    metric = explore_conf_dict['metric']
    algorithm = explore_conf_dict['algorithm']
    quantiles = pd.Series(explore_conf_dict['quantiles'])
    sample_miss_cutoff = explore_conf_dict['sample_miss_cutoff']
    variable_miss_cutoff = explore_conf_dict['variable_miss_cutoff']
    variable_zero_cutoff = explore_conf_dict['variable_zero_cutoff']
    max_nFactorLevels = explore_conf_dict['max_nFactorLevels']
    nbins = explore_conf_dict['nbins']
    truncation_cutoff = explore_conf_dict['truncation_cutoff']
    breaks_zero_cutoff = explore_conf_dict['breaks_zero_cutoff']
    iv_cutoff = explore_conf_dict['iv_cutoff']
    collinearity_cutoff = explore_conf_dict['collinearity_cutoff']
    unbalanced_cutoff = explore_conf_dict['unbalanced_cutoff']
    onehot = explore_conf_dict['onehot']

    if isinstance(metric, str):
        audit_result_dict['The class of sample_miss_cutoff is character'] = True
    else:
        audit_result_dict['The class of sample_miss_cutoff is character'] = False
    if metric in ["ks", "auc"]:
        audit_result_dict['metric is ks or auc'] = True
    else:
        audit_result_dict['metric is ks or auc'] = False

    if isinstance(algorithm, str):
        audit_result_dict['The class of algorithm is character'] = True
    else:
        audit_result_dict['The class of algorithm is character'] = False
    if algorithm in ["glmnet", "cart"]:
        audit_result_dict['algorithm is glmnet or cart'] = True
    else:
        audit_result_dict['algorithm is glmnet or cart'] = False

    if quantiles[(quantiles >= 0) & (quantiles <= 1)].count() == quantiles.count():
        audit_result_dict['The value of quantiles lie in [0,1]'] = True
    else:
        audit_result_dict['The value of quantiles lie in [0,1]'] = False

    if isinstance(sample_miss_cutoff, float):
        audit_result_dict['The class of sample_miss_cutoff is numeric'] = True
    else:
        audit_result_dict['The class of sample_miss_cutoff is numeric'] = False
    if ((sample_miss_cutoff < 0.6) | (sample_miss_cutoff > 1)):
        audit_result_dict['sample_miss.cutoff lie in [0.6,1]'] = False
    else:
        audit_result_dict['sample_miss.cutoff lie in [0.6,1]'] = True

    if isinstance(variable_miss_cutoff, float):
        audit_result_dict['The class of variable_miss_cutoff is numeric'] = True
    else:
        audit_result_dict['The class of variable_miss_cutoff is numeric'] = False
    if ((variable_miss_cutoff < 0.6) | (variable_miss_cutoff > 1)):
        audit_result_dict['variable_miss.cutoff is lie in [0.6,1]'] = False
    else:
        audit_result_dict['variable_miss.cutoff is lie in [0.6,1]'] = True

    if isinstance(variable_zero_cutoff, float):
        audit_result_dict['The class of variable_zero_cutoff is numeric'] = True
    else:
        audit_result_dict['The class of variable_zero_cutoff is numeric'] = False
    if ((variable_zero_cutoff < 0.6) | (variable_zero_cutoff > 1)):
        audit_result_dict['variable_zero_cutoff is lie in [0.6,1]'] = False
    else:
        audit_result_dict['variable_zero_cutoff is lie in [0.6,1]'] = True

    if isinstance(max_nFactorLevels, int):
        audit_result_dict['The class of max.nFactorLevels is int'] = True
    else:
        audit_result_dict['The class of max.nFactorLevels is int'] = False
    if ((max_nFactorLevels < 2) | (max_nFactorLevels > 2000)):
        audit_result_dict['max_nFactorLevels is lie in [2,2000]'] = False
    else:
        audit_result_dict['max_nFactorLevels is lie in [2,2000]'] = True

    if isinstance(nbins, int):
        audit_result_dict['The class of nbins is int'] = True
    else:
        audit_result_dict['The class of nbins is int'] = False
    if ((nbins < 3) | (nbins > 20)):
        audit_result_dict['nbins is lie in [3,20]'] = False
    else:
        audit_result_dict['nbins is lie in [3,20]'] = True

    if isinstance(truncation_cutoff, int):
        audit_result_dict['The class of truncation_cutoff is int'] = True
    else:
        audit_result_dict['The class of truncation_cutoff is int'] = False
    if ((truncation_cutoff < 2) | (truncation_cutoff > 20)):
        audit_result_dict['truncation_cutoff is lie in [2,20]'] = False
    else:
        audit_result_dict['truncation_cutoff is lie in [2,20]'] = True

    if isinstance(breaks_zero_cutoff, float):
        audit_result_dict['The class of breaks_zero_cutoff is numeric'] = True
    else:
        audit_result_dict['The class of breaks_zero_cutoff is numeric'] = False
    if ((breaks_zero_cutoff < 0.1) | (breaks_zero_cutoff > 1)):
        audit_result_dict['breaks_zero_cutoff is lie in [0.1,1]'] = False
    else:
        audit_result_dict['breaks_zero_cutoff is lie in [0.1,1]'] = True

    if isinstance(iv_cutoff, float):
        audit_result_dict['The class of iv_cutoff is numeric'] = True
    else:
        audit_result_dict['The class of iv_cutoff is numeric'] = False
    if ((iv_cutoff < 0) | (iv_cutoff > 1)):
        audit_result_dict['iv_cutoff is lie in [0,1]'] = False
    else:
        audit_result_dict['iv_cutoff is lie in [0,1]'] = True

    if isinstance(collinearity_cutoff, float):
        audit_result_dict['The class of collinearity_cutoff is numeric'] = True
    else:
        audit_result_dict['The class of collinearity_cutoff is numeric'] = False
    if ((collinearity_cutoff < 0) | (collinearity_cutoff > 1)):
        audit_result_dict['collinearity_cutoff is lie in [0,1]'] = False
    else:
        audit_result_dict['collinearity_cutoff is lie in [0,1]'] = True

    if isinstance(unbalanced_cutoff, int):
        audit_result_dict['The class of unbalanced_cutoff is int'] = True
    else:
        audit_result_dict['The class of unbalanced_cutoff is int'] = False
    if ((unbalanced_cutoff < 1) | (unbalanced_cutoff > 10)):
        audit_result_dict['unbalanced_cutoff is lie in [1,10]'] = False
    else:
        audit_result_dict['unbalanced_cutoff is lie in [1,10]'] = True

    if isinstance(onehot, bool):
        audit_result_dict['The class of onehot is bool'] = True
    else:
        audit_result_dict['The class of onehot is bool'] = False

    return audit_result_dict


def check_id(data_df, id_name, stage=None):
    """
    check the dataframe: 
    1. if id col exist
    2. when id col exists, whether each row of the id col is unique
    3. if null value exists in id col
    :param: data_df: a data frame 
    :param: id_name: a specified id name
    :param: stage: explore or act
    :return: audit result, dict.
    """
    audit_result_dict = dict()
    if isinstance(id_name, str):
        audit_result_dict['The class of id_name is character'] = True
    else:
        audit_result_dict['The class of id_name is character'] = False

    if data_df.columns.values.tolist().count(id_name) == 1:
        audit_result_dict["id in data and id is unique variable"] = True
    else:
        audit_result_dict["id in data and id is unique variable"] = False

    if stage != 'explore':
        unique_lst = list(data_df[id_name].unique())
        if data_df.shape[0] == len(unique_lst):
            audit_result_dict['no repeated id'] = True
        else:
            audit_result_dict['no repeated id'] = False

        if data_df[id_name].isnull().sum() == 0:
            audit_result_dict['no null id'] = True
        else:
            audit_result_dict['no null id'] = False
        del unique_lst

    return audit_result_dict


def check_target(data_df, target_name):
    """
    check the dataframe: 
    1. if label col exist
    2. if null value exists in label col
    3. if target only contain '1', '0'
    4. if '1' represent minor level
    :param: data_df: a data frame
    :param: target_name: a specified target name
    :return: audit result, dict.
    """
    audit_result_dict = dict()
    if isinstance(target_name, str):
        audit_result_dict['The class of target_name is character'] = True
    else:
        audit_result_dict['The class of target_name is character'] = False

    if data_df.columns.values.tolist().count(target_name) == 1:
        audit_result_dict["target in data and target is unique variable"] = True
    else:
        audit_result_dict["target in data and target is unique variable"] = False

    if data_df[target_name].isnull().sum() == 0:
        audit_result_dict['The target variable of the training data have no missing value'] = True
    else:
        audit_result_dict['The target variable of the training data have no missing value'] = False

    levels_amount = cal_vars_levels_amount(data_df, target_name)
    if set(levels_amount.keys()) == set([0, 1]):
        audit_result_dict['only 1 and 0 in target'] = True

        if levels_amount[1] < levels_amount[0]:
            audit_result_dict['1 is minority class'] = True
        else:
            audit_result_dict['1 is minority class'] = False
    else:
        audit_result_dict['only 1 and 0 in target'] = False

    del levels_amount

    return audit_result_dict


def check_type_dict(data_type_dict, target_name=None):
    """
    check type dictionary: 
    1. if it only contain 'numeric', 'factor'
    2. if id and target in the dictionary
    :param: data_type_dict: a map of variables and their types, dict.
    :param: target_name: target name
    :return: audit result, dict.
    """

    audit_result_dict = dict()
    if set(set(data_type_dict.values())) == set(['numeric', 'factor']):
        audit_result_dict['only numeric and factor contained in the dictionary'] = True
    else:
        audit_result_dict['only numeric and factor contained in the dictionary'] = False

    if target_name is not None:
        if target_name in list(data_type_dict.keys()):
            audit_result_dict['target in data type dictionary'] = True
        else:
            audit_result_dict['target in data type dictionary'] = False

    return audit_result_dict


def check_vars(data_df, vars_lst):
    """
    check if variables needed are in the dataframe
    :param data_df: target dataframe
    :param vars_lst: needed vars, list of str
    :return: audit result, dict.
    """
    audit_result_dict = dict()
    data_difference_set = list(set(vars_lst).difference(set(data_df.columns.tolist())))
    if len(data_difference_set) > 0:
        audit_result_dict['The variables of dictionary not missing in data'] = False
    else:
        audit_result_dict['The variables of dictionary not missing in data'] = True

    return audit_result_dict


def check_data_vol(data_df, row_low_limit, col_low_limit):
    """
    check if data set is big enough
    :param: data_df: a data frame
    :param: row_low_limit: lower limit of number of rows
    :param: col_low_limit: lower limit of number of cols
    :return: audit result, dict.
    """
    audit_result_dict = dict()
    df_shape = data_df.shape
    if df_shape[1] > col_low_limit:
        audit_result_dict['vars is too small'] = True
    else:
        audit_result_dict['vars is too small'] = False

    if df_shape[0] > row_low_limit:
        audit_result_dict['samples is too small'] = True
    else:
        audit_result_dict['samples is too small'] = False

    return audit_result_dict




