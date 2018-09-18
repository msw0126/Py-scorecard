import math
from Utilities.DataProcess import *
import datetime
from Utilities.LogTools import *
import numpy as np


def cal_woe(groupby_df, pos_sum, neg_sum):
    """
    calculate woe
    :param groupby_df: grouped dataframe ( group target 1\0 by levels in variable)
    :param pos_sum: total label 1 amount, int.
    :param neg_sum: total label 0 amount, int.
    :return: a dict with woe infomation of each level
    """
    groupby_df['pos_ratio'] = groupby_df['npos'] * 1.0 / pos_sum
    groupby_df['nneg'] = groupby_df['nrow'] - groupby_df['npos']
    groupby_df['neg_ratio'] = groupby_df['nneg'] * 1.0 / neg_sum
    groupby_df['prevalence'] = groupby_df['npos'] * 1.0 / groupby_df['nrow']
    correction_factor = 0.00001
    groupby_df['woe'] = np.log((groupby_df['pos_ratio'] + correction_factor)\
                               / (groupby_df['neg_ratio'] + correction_factor))
    var_woe_dict = dict([(str(i[1]['var_level']), i[1]) for i in groupby_df.to_dict(orient='index').items()])

    return var_woe_dict


def univar_groupby(data_df, target_name, var_name):
    """
    group variable level to get label 1 and label 0 amount, in order to calculate woe and iv.
    :param data_df: target dataframe
    :param target_name: label variable name
    :param var_name: variable that to group its levels
    :return: a grouped dataframe
    """
    groupby_df = data_df[[var_name, target_name]]\
        .groupby(by=var_name)\
        .agg({target_name: ['count', 'sum']})

    groupby_df.columns = groupby_df.columns.map('_'.join)
    groupby_df = groupby_df.reset_index().rename(columns={var_name: 'var_level',
                                                          target_name + '_sum': 'npos',
                                                          target_name + '_count': 'nrow'})

    return groupby_df


def process_unique_numeric(data_df, var_name, target_name, pos_sum, neg_sum, null_number):
    """
    process numeirc variable that have unique values less than nbins: cut and calculate woe 
    :param data_df: target dataframe
    :param var_name: variable name
    :param target_name: label name
    :param pos_sum: total label 1 amount 
    :param neg_sum: total label 0 amount
    :return: a list of breaks, a list of labels of each bin, and a woe result dict
    """
    # get breaks
    breaks_lst = list(pd.unique(data_df[var_name].dropna()))

    breaks_lst.sort()
    breaks_lst.append(breaks_lst[-1] + 1)
    # set labels for bins
    if null_number is None:
        breaks_labels = [str(i + 1) for i in range(len(breaks_lst)-1)]
    else:
        breaks_lst.insert(0, null_number)
        data_df = fill_missing(data_df, {var_name: null_number})
        breaks_labels = [str(i) for i in range(len(breaks_lst)-1)]

    # extend first and last limits
    tmp_breaks = breaks_lst[:]
    # conduct cut action
    data_df[var_name] = pd.cut(data_df[var_name], bins=tmp_breaks, labels=breaks_labels,
                               include_lowest=True, right=False, precision=3)
    # conduct group by levels
    groupby_df = univar_groupby(data_df, target_name, var_name)
    # cal woe
    var_woe_dict = cal_woe(groupby_df, pos_sum, neg_sum)

    del groupby_df
    gc.collect()

    data_df[var_name] = pd.to_numeric(data_df[var_name])

    return data_df, breaks_lst, breaks_labels, var_woe_dict


def cut_zeros_numeric(data_df, var_name, nbins, target_name, pos_sum, neg_sum, null_number):
    """
    process numeirc variable that have too much zeros: cut and calculate woe
    :param data_df: target dataframe
    :param var_name: variable name
    :param nbins: number of bins that want to cut
    :param target_name: label name
    :param pos_sum: total label 1 amount 
    :param neg_sum: total label 0 amount
    :param null_number:
    :return: a list of breaks, a list of labels of each bin, and a woe result dict
    """
    # get breaks
    percentiles = [i * 1.0 / (nbins - 1) for i in range(nbins)]
    breaks_lst = cal_quantiles(data_df, var_name,
                               percentiles=percentiles, omit_zero=True, rand_num=None)
    breaks_lst.append(0)
    breaks_lst = sorted(list(set(breaks_lst)))

    # set labels for cutted bin
    if null_number is None:
        breaks_labels = [str(i + 1) for i in range(len(breaks_lst) - 1)]
    else:
        breaks_lst.insert(0, null_number)
        data_df = fill_missing(data_df, {var_name: null_number})
        breaks_labels = [str(i) for i in range(len(breaks_lst) - 1)]

    breaks_lst[-1] = breaks_lst[-1] + 1
    # extend first and last limits
    tmp_breaks = breaks_lst[:]
    tmp_breaks[0] = tmp_breaks[0] - 1
    tmp_breaks[-1] = tmp_breaks[-1] + 1

    # conduct cut action
    data_df[var_name] = pd.cut(data_df[var_name], bins=tmp_breaks, labels=breaks_labels,
                               include_lowest=True, right=False, precision=3)

    # conduct group by levels
    groupby_df = univar_groupby(data_df, target_name, var_name)
    # cal woe
    var_woe_dict = cal_woe(groupby_df, pos_sum, neg_sum)

    del groupby_df
    gc.collect()

    data_df[var_name] = pd.to_numeric(data_df[var_name])

    return data_df, breaks_lst, breaks_labels, var_woe_dict


def cut_normal_numeric(data_df, var_name, breaks_lst, target_name, pos_sum, neg_sum, null_number):
    """
    process normal numeirc variable : cut and calculate woe
    :param data_df: target dataframe
    :param var_name: variable name
    :param breaks_lst: a list of break values to cut
    :param target_name: label name
    :param pos_sum: total label 1 amount 
    :param neg_sum: total label 0 amount
    :param null_number: 
    :return: a list of labels of each bin, and a woe result dict
    """
    # set labels for cutted bin
    if null_number is None:
        breaks_labels = [str(i + 1) for i in range(len(breaks_lst) - 1)]
    else:
        breaks_labels = [str(i) for i in range(len(breaks_lst) - 1)]

    # extend first and last limits
    tmp_breaks = breaks_lst[:]
    tmp_breaks[0] = tmp_breaks[0] - 1
    tmp_breaks[-1] = tmp_breaks[-1] + 1
    # conduct cut action
    data_df[var_name] = pd.cut(data_df[var_name], bins=tmp_breaks, labels=breaks_labels,
                               include_lowest=True, right=False, precision=3)

    # conduct group by levels
    groupby_df = univar_groupby(data_df, target_name, var_name)
    # cal woe
    var_woe_dict = cal_woe(groupby_df, pos_sum, neg_sum)

    del groupby_df
    gc.collect()

    data_df[var_name] = pd.to_numeric(data_df[var_name])

    return data_df, breaks_labels, var_woe_dict


def cut_numeric_main(data_df, numeric_vars, target_name, pos_sum, neg_sum,
                     nbins=None, truncation_cutoff=None, breaks_zero_cutoff=None):
    """
    cut numeric variables into n bins and calculate their woes
    :param data_df: target dataframe
    :param numeric_vars: numeric variables, list of str.
    :param target_name: label name
    :param pos_sum: total label 1 amount 
    :param neg_sum: total label 0 amount
    :param nbins: bin number for cutting, int.
    :param truncation_cutoff: cutoff value for truncation (float or int), 
                              truncate when value > 0.99_quantile_value * truncation_cutoff
                              or < 0.01_quantile_value / truncation_cutoff .
    :param breaks_zero_cutoff: threshold of zeros ratio, if > threshold: set 0 as a break value.
    :return: a cutted dataframe, a map of numeirc vars and their breaks configuration
    """
    nrows = data_df.shape[0]
    numeric_breaks_dict = dict()

    # get numeric variables' missing counts and zero amounts
    statistics_dict = get_describe_result(data_df, numeric_vars)
    normal_cut_vars = list()
    # convert target var to numeirc, ready for compute woe
    data_df[target_name] = pd.to_numeric(data_df[target_name])
    i = 0
    woe_dict = dict()
    for var in numeric_vars:
        i += 1
        var_statistics_dict = statistics_dict[var]
        na_num = nrows - var_statistics_dict['count']

        if na_num > 0:
            null_number = var_statistics_dict['min'] - 1
        else:
            null_number = None

        # judge whether to cut (only if unique number > nbin)
        # if nrows less than 100000, calculate all,
        # else first try first 100000 rows, if unique not enough, then rand sample 100000, and try again
        if nrows <= 10000:
            uniques = data_df[var].nunique()
        else:
            uniques = data_df.loc[:10000, var].nunique()

        if uniques <= nbins:
            if nrows > 100000:
                tmp_df = random_sampling(data_df[var], sub_num=100000)
                uniques = tmp_df.nunique()
            # if unique values less than nbins, cut variables directely by unique values
            if uniques <= nbins:
                data_df, breaks_lst, breaks_labels, var_woe_dict \
                    = process_unique_numeric(data_df, var, target_name, pos_sum, neg_sum, null_number)
                woe_dict[var] = var_woe_dict
        if uniques > nbins:

            zero_num = data_df[data_df[var] == 0].shape[0]
            # if zeros ratio bigger than cutoff value, add zero into breaks, then cut
            if zero_num * 1.0 / (nrows - na_num) > breaks_zero_cutoff:
                data_df, breaks_lst, breaks_labels, var_woe_dict \
                    = cut_zeros_numeric(data_df, var, nbins, target_name, pos_sum, neg_sum, null_number)
                woe_dict[var] = var_woe_dict
            else:
                normal_cut_vars.append(var)
                breaks_labels, breaks_lst = list(), list()
        # store cut information
        numeric_breaks_dict[var] = dict({'breaks_labels': breaks_labels,
                                         'breaks_lst': breaks_lst,
                                         'null_number': null_number})
    # normal cut
    if len(normal_cut_vars) > 0:
        percentiles = [j * 1.0 / nbins for j in range(nbins + 1)]
        var_breaks_dict = cal_vars_quantiles(data_df, normal_cut_vars, percentiles=percentiles)
        for var in normal_cut_vars:
            breaks_lst = sorted(list(set(var_breaks_dict[var])))
            null_number = numeric_breaks_dict[var]['null_number']
            if null_number is not None:
                breaks_lst.insert(0, null_number)
                data_df = fill_missing(data_df, {var: null_number})
            breaks_lst[-1] = breaks_lst[-1] + 1

            data_df, breaks_labels, var_woe_dict \
                = cut_normal_numeric(data_df, var, breaks_lst, target_name, pos_sum, neg_sum, null_number)
            woe_dict[var] = var_woe_dict
            numeric_breaks_dict[var]['breaks_labels'] = breaks_labels
            numeric_breaks_dict[var]['breaks_lst'] = breaks_lst

    return data_df, numeric_breaks_dict, woe_dict


def merge_minor_levels(data_df, factor_vars, merge_threshould=0.1):
    """
    
    :param data_df: 
    :param factor_vars: 
    :param merge_threshould: 
    :return: 
    """
    nrow = data_df.shape[0]
    merge_dict = {}
    for var in factor_vars:
        level_amount_dict = cal_vars_levels_amount(data_df, var)
        level_amount_items = sorted(level_amount_dict.items(), key=lambda x: x[1])
        cum = 0
        merge_levels = []
        for i in level_amount_items:
            cum += i[1]
            if (cum * 1.0 / nrow) >= merge_threshould:
                break
            merge_levels.append(i[0])

        if len(merge_levels) > 1:
            data_df[var] = data_df[var].replace(to_replace=merge_levels,
                                                value='|'.join(merge_levels))

        merge_dict[var] = merge_levels

    return data_df, merge_dict


def process_factor_vars(data_df, factor_vars, target_name, pos_sum, neg_sum):
    """
    process factor variable: calculate woe
    :param data_df: target dataframe
    :param factor_vars: factor variables, a list of str.
    :param target_name: label name
    :param pos_sum: total label 1 amount 
    :param neg_sum: total label 0 amount
    :return: a dict of factor variables and their woe information
    """
    data_df[target_name] = pd.to_numeric(data_df[target_name])
    for var in factor_vars:
        data_df[var] = data_df[var].cat.add_categories(['null_flag'])
    data_df[factor_vars] = data_df[factor_vars].fillna(value='null_flag')
    data_df, merge_dict = merge_minor_levels(data_df, factor_vars, merge_threshould=0.2)
    woe_dict = dict()
    for var in factor_vars:
        groupby_df = univar_groupby(data_df, target_name, var)
        var_woe_dict = cal_woe(groupby_df, pos_sum, neg_sum)
        del groupby_df
        gc.collect()
        woe_dict[var] = var_woe_dict

    return data_df, woe_dict, merge_dict


def chi2_test_2x2(data_lst):
    """
    conduct a 2*2 Chi-square test
    :param data_lst: a list of two lists like [[a, b], [c, d]]
    :return: a Chi2 value(float) and its p value region(a two value list)
    """
    # chi2 value - p table for 1 degree
    chi2_table_1df = {0.995: 0.0000393, 0.975: 0.000982, 0.20: 1.642, 0.10: 2.706, 0.05: 3.841,
                      0.025: 5.024, 0.02: 5.412, 0.01: 6.635, 0.005: 7.879, 0.002: 9.550, 0.001: 10.828}
    a = data_lst[0][0]
    b = data_lst[0][1]
    c = data_lst[1][0]
    d = data_lst[1][1]
    abcd_sum = a + b + c + d
    if len(list(filter(lambda x: x == 0, [a, b, c, d]))) == 1:
        # it means two sample are obviously different
        chi2_value = 100
    elif a > 0 and b == 0 and c > 0 and d == 0:
        # it means two sample are similar
        chi2_value = 0
    elif a == 0 and b > 0 and c == 0 and d > 0:
        # it means two sample are similar
        chi2_value = 0
    elif len(list(filter(lambda x: x == 0, [a, b, c, d]))) in [3, 2]:
        # it means two sample are obviously different
        chi2_value = 100
    elif len(list(filter(lambda x: x == 0, [a, b, c, d]))) == 4:
        # it means two sample are similar
        chi2_value = 0
    elif a < 5 or b < 5 or c < 5 or d < 5:
        # correction chi2 test
        correction_factor = 0.1
        chi2_value = (1.0 * (abs(a * d - b * c) - 1.0 * abcd_sum / 2)**2 * abcd_sum + correction_factor) \
                     / ((a + b) * (c + d) * (a + c) * (b + d) + correction_factor)
    else:
        # normal chi2 test
        correction_factor = 0.1
        chi2_value = (1.0 * (a * d - b * c)**2 * abcd_sum + correction_factor) \
                     / ((a + b) * (c + d) * (a + c) * (b + d) + correction_factor)

    p_value = [0.0, 1.0]
    for alpha in sorted(list(chi2_table_1df.keys())):
        if chi2_value > chi2_table_1df[alpha]:
            p_value[1] = alpha
            break
        elif chi2_value == chi2_table_1df[alpha]:
            p_value[1] = alpha
            p_value[0] = alpha
            break
        elif chi2_value < chi2_table_1df[alpha]:
            p_value[0] = alpha

    return chi2_value, p_value


def bins_merge(data_df, woe_dict, bin_dict, p_threshold=0.2):
    """
    combine bins based on Chi2 test between adjacent bins.
    :param data_df:
    :param woe_dict: a dict with woe info of each variable
    :param bin_dict: a dict with bin info of each variable
    :param p_threshold: the alpha value to decision whether combine two bins
    :return: a dict with woe info and a dict with bin info after combination
    """

    for col in woe_dict.keys():
        col_woe_dict = woe_dict[col]
        bin_labels = bin_dict[col][0]
        bin_breaks = bin_dict[col][1]

        merge_flag = False
        for i in range(len(bin_labels) - 1):
            if bin_labels[i] == '0':
                continue

            label_1 = bin_labels[i]
            data_lst = list()
            data_lst.append([1.0 * col_woe_dict[label_1]['npos'],
                             1.0 * col_woe_dict[label_1]['nneg']])
            label_2 = bin_labels[i + 1]
            data_lst.append([1.0 * col_woe_dict[label_2]['npos'],
                             1.0 * col_woe_dict[label_2]['nneg']])
            # calculate chi2 value
            chi2_value, p_value = chi2_test_2x2(data_lst)

            if p_value[0] >= p_threshold:
                # combine bin info of these two bins
                label_1_dict = col_woe_dict[label_1]
                label_2_dict = col_woe_dict[label_2]
                tmp_dict = dict({'nrow': label_1_dict['nrow'] + label_2_dict['nrow'],
                                 'npos': label_1_dict['npos'] + label_2_dict['npos'],
                                 'nneg': label_1_dict['nneg'] + label_2_dict['nneg'],
                                 'pos_ratio': label_1_dict['pos_ratio'] + label_2_dict['pos_ratio'],
                                 'neg_ratio': label_1_dict['neg_ratio'] + label_2_dict['neg_ratio']})

                tmp_dict['prevalence'] = tmp_dict['npos'] * 1.0 / tmp_dict['nrow']

                # correction for avoiding 0 division and log(0)
                correction_factor = 0.00001
                correction = (tmp_dict['pos_ratio'] == 0 or tmp_dict['neg_ratio'] == 0)
                tmp_dict['woe'] = math.log((tmp_dict['pos_ratio'] + correction * correction_factor)
                                           / (tmp_dict['neg_ratio'] + correction * correction_factor))

                # updata woe info dict
                del col_woe_dict[label_2], col_woe_dict[label_1]
                gc.collect()

                bin_labels[i + 1] = label_1 + '_' + label_2
                col_woe_dict[bin_labels[i + 1]] = tmp_dict

                bin_labels[i] = None
                bin_breaks[i + 1] = None

        # update
        bin_breaks = [i for i in bin_breaks if i is not None]
        bin_labels_old = [i for i in bin_labels if i is not None]

        if '0' in bin_labels:
            bin_labels = [str(i) for i in range(len(bin_breaks) - 1)]
        else:
            bin_labels = [str(i + 1) for i in range(len(bin_breaks) - 1)]

        for i in range(len(bin_labels_old)):
            if '_' in bin_labels_old[i]:
                merge_bins = [int(j) for j in bin_labels_old[i].split('_')]
                data_df[col] = data_df[col].replace(to_replace=merge_bins,
                                                    value=int(bin_labels[i]))
            else:
                if str(bin_labels[i]) != str(bin_labels_old[i]):
                    data_df[col] = data_df[col].replace(to_replace=int(bin_labels_old[i]),
                                                        value=int(bin_labels[i]))

        new_col_woe_dict = {}
        for i in range(len(bin_labels_old)):
            new_col_woe_dict[bin_labels[i]] = col_woe_dict[bin_labels_old[i]]

        woe_dict[col] = new_col_woe_dict

        bin_dict[col][0] = bin_labels
        bin_dict[col][1] = bin_breaks

    return data_df, woe_dict, bin_dict


def cal_iv(woe_dict):
    """
    calculate iv value
    :param woe_dict: target dataframe
    :return: a map of vars and their iv values (dict), a list of iv value for output, a woe dict
    """
    iv_dict = dict()
    iv_list = list([['variable', 'iv']])
    woe_list = list([['variable', 'bin', 'woe']])
    for var in woe_dict.keys():
        iv = 0
        tmp_dict = woe_dict[var]
        for level in tmp_dict.keys():
            woe_list.append([var, level, tmp_dict[level]['woe']])
            iv += (tmp_dict[level]['pos_ratio'] - tmp_dict[level]['neg_ratio']) * tmp_dict[level]['woe']
        iv_dict[var] = iv
        iv_list.append([var, iv])
    return iv_dict, iv_list


def find_vars_under_iv_cutoff(iv_dict, iv_cutoff=None):
    """
    find variables with theirs iv values smaller than cutoff value
    :param iv_dict: a map of vars and their iv values, dict.
    :param iv_cutoff: cutoff value of iv, float.
    :return: a list of variables to remove, list of str
    """
    remove_list = list()
    for var in iv_dict.keys():
        if iv_dict[var] < iv_cutoff:
            remove_list.append(var)

    return remove_list


@time_decrator('binning')
def binning_analysis_main(data_obj, explore_conf_dict):
    """
    main function of binning analysis: cut numeric variables, calculate each variables' woe and iv
    :param data_obj: target data object
    :param explore_conf_dict: configuration information for explore
    :return: data object after analysis and iv cutting off
    """
    # set basic information
    numeric_vars = data_obj['info']['numeric_vars']
    factor_vars = data_obj['info']['factor_vars']
    target_name = explore_conf_dict['target_varname']
    pos_sum = data_obj['df'][data_obj['df'][target_name] == 1].shape[0]
    neg_sum = data_obj['df'].shape[0] - pos_sum
    nbins = explore_conf_dict['nbins']
    truncation_cutoff = explore_conf_dict['truncation_cutoff']
    breaks_zero_cutoff = explore_conf_dict['breaks_zero_cutoff']

    # break numeric vars and cal woes
    data_obj['df'], numeric_breaks_dict, numeirc_woe_dict = \
        cut_numeric_main(data_obj['df'], numeric_vars, target_name, pos_sum, neg_sum,
                         nbins, truncation_cutoff, breaks_zero_cutoff)

    bin_dict = dict([(var, [numeric_breaks_dict[var]['breaks_labels'][:],
                            numeric_breaks_dict[var]['breaks_lst'][:]])
                     for var in numeric_breaks_dict.keys()])
    data_obj['df'], numeirc_woe_dict, bin_dict = bins_merge(data_obj['df'], numeirc_woe_dict, bin_dict)

    for var in numeric_breaks_dict.keys():
        numeric_breaks_dict[var]['breaks_labels'] = bin_dict[var][0]
        numeric_breaks_dict[var]['breaks_lst'] = bin_dict[var][1]
    data_obj['info']['numeric_breaks_dict'] = numeric_breaks_dict

    # cal factor vars' woes, ready for calculate iv
    data_obj['df'], factor_woe_dict, merge_dict \
        = process_factor_vars(data_obj['df'], factor_vars, target_name, pos_sum, neg_sum)
    data_obj['info']['factor_merge_dict'] = merge_dict

    woe_dict = numeirc_woe_dict
    woe_dict.update(factor_woe_dict)

    # calculate iv
    iv_dict, iv_list = cal_iv(woe_dict)
    data_obj['info']['iv_dict'] = iv_dict
    data_obj['info']['iv_list'] = iv_list
    data_obj['info']['woe_dict'] = woe_dict
    # remove vars by iv value
    iv_cutoff = explore_conf_dict['iv_cutoff']
    if iv_cutoff:
        remove_vars = find_vars_under_iv_cutoff(iv_dict, iv_cutoff)
        data_obj['info']['remove_dict']['iv_vars'] = remove_vars
        data_obj['info']['numeric_vars'] = list(set(data_obj['info']['numeric_vars']) - set(remove_vars))
        data_obj['info']['factor_vars'] = list(set(data_obj['info']['factor_vars']) - set(remove_vars))
        data_obj['info']['remove_vars'] = list(set(data_obj['info']['remove_vars']) | set(remove_vars))
        data_obj['df'] = del_vars(data_obj['df'], remove_vars)

    return data_obj

