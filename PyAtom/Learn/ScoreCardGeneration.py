# -*- coding:utf-8 -*-
import re
import time

import pandas as pd
import numpy as np
import json


def card_to_json(score_card):
    """
    评分卡转为json格式
    :param score_card:
    :return:
    """
    scorecard_variable = score_card['variable'].unique()
    score_card_dict = {}
    for var_name in scorecard_variable:
        tmp_dict = dict()
        var_score_card = score_card[score_card['variable'].isin([var_name])]
        for x in var_score_card['bin'].values.tolist():
            tmp_dict.setdefault("bin", []).append(str(x))
        for x in var_score_card['points'].values.tolist():
            tmp_dict.setdefault("points", []).append(x)
        score_card_dict[var_name] = tmp_dict
    return json.dumps(score_card_dict)


def get_a_b(points0, odds0, pdo):
    """
    计算补偿和刻度
    """
    if pdo > 0:
        b = pdo / np.log(2)
    else:
        b = -pdo / np.log(2)
    a = points0 + b * np.log(odds0)
    return a, b


def get_score_card(bins_df, model, xcolumns, factor_vars_variable_and_lable, points0=600, odds0=1/19, pdo=50,basepoints_eq0=False):
    """
    得到评分卡
    :param bins: 分箱字典
    :param model: 训练模型
    :param xcolumns: 训练数据变量名
    :param points0:评分刻度,默认600
    :param odds0:坏好比，坏好比越高，score越低
    :param pdo:比率翻倍的分值
    :param basepoints_eq0:如果为False会单独显示一行basepoints，如果为True会把基础分会均分到其他变量上
    :return:
    """
    # R手把手计算方法
    # coe = model.coef_
    # print("len(coe):{}".format(len(coe)))
    # b = 20 / np.log(2)
    # a = 600 - 20 * np.log(2.5) / np.log(2)
    # basepoints = a + b * coe[0][0]
    # print(base_points)
    if isinstance(bins_df, dict):
        bins_df = pd.concat(bins_df, ignore_index=True)
    # 计算补偿和刻度
    a, b = get_a_b(points0, odds0, pdo)
    # 逻辑回归的coe系数(model.coef_[0]输出权重weight)
    coef_df = pd.Series(model.coef_[0], index=np.array(xcolumns))#.loc[lambda x: x != 0]
    # 基础分(model.intercept_[0]输出偏置bias)
    basepoints = a - b * model.intercept_[0]
    card = {}
    card['basepoints'] = pd.DataFrame({'variable': "basepoints", 'bin': np.nan, 'points': round(basepoints)},
                                      index=np.arange(1))
    for i in coef_df.index:
        card[i] = bins_df.loc[bins_df['variable'] == i, ['variable', 'bin', 'woe']] \
            .assign(points=lambda x: round(-b * x['woe'] * coef_df[i]))[['variable', 'bin', "points"]]
    score_card_df = pd.concat(card, ignore_index=True)
    score_card_df['variable'] = score_card_df['variable'].apply(lambda x: re.sub('_\d+$', '', x))
    for label in factor_vars_variable_and_lable:
        label = str(label).replace("|", "\|")
        score_card_df['variable'] = score_card_df['variable'].apply(lambda x: re.sub('{}$'.format(label), '', x))
    return score_card_df


def scorecard_apply(train_data, score_card, preprocess_result):
    """
    根据评分卡得到每个样本的分值
    """
    numeric_vars = preprocess_result['numeric_vars']
    factor_vars = preprocess_result['factor_vars']
    card_df = score_card
    # 得到评分卡的所有变量
    score_card_var_names = card_df.loc[card_df.variable != 'basepoints', 'variable'].unique()
    # 得到基础分值
    base_points = card_df[card_df.variable == "basepoints"].points.values[0]
    score_df = pd.DataFrame()
    # 根据
    for score_card_var_name in score_card_var_names:
        df = pd.DataFrame()
        # 定义单一变量相关的评分卡与data
        card_var_property = card_df.loc[card_df['variable'] == score_card_var_name].reset_index(drop=True)
        data_card_var = train_data[[score_card_var_name]]
        # numeric类型变量处理
        if score_card_var_name in numeric_vars:
            for bin in card_var_property['bin']:
                # print(type(bin))
                if "null_flag" not in bin:
                    min_val = float(re.search(r'.*\[(.*),(.*)\].*', bin).group(1))
                    max_val = float(re.search(r'.*\[(.*),(.*)\].*', bin).group(2))
                    points = card_df[(card_df.variable == score_card_var_name) & (card_df.bin == bin)].points.values[0]
                    format = lambda x: points if max_val > x >= min_val else 0
                    data_card_var_r = data_card_var.applymap(format)
                else:
                    points = card_df[(card_df.variable == score_card_var_name) & (card_df.bin == bin)].points.values[0]
                    format = lambda x: points if np.isnan(x) else 0
                    data_card_var_r = data_card_var.applymap(format)
                    # print(data_card_var_r.head(10))
                df = pd.concat([df, data_card_var_r], axis=1)
            df = df.T
            # print(df)
            # 变量不同index的分值
            var_name_points = df.apply(lambda x: x.sum(), axis=0).to_frame(name=score_card_var_name + "_points")
            # print(var_name_points)
        else:
            for bin in card_var_property['bin']:
                points = card_df[(card_df.variable == score_card_var_name) & (card_df.bin == bin)].points.values[0]
                if "null_flag" not in bin:
                    format = lambda x: points if str(x) in bin else 0
                else:
                    format = lambda x: points if pd.isna(x) else 0
                data_card_var_r = data_card_var.applymap(format)
                df = pd.concat([df, data_card_var_r], axis=1)
            df = df.T
            # 变量不同index的分值
            var_name_points = df.apply(lambda x: x.sum(), axis=0).to_frame(name=score_card_var_name + "_points")
        score_df = pd.concat([score_df, var_name_points], axis=1)
    score_df['score'] = score_df.apply(lambda x: x.sum() + base_points, axis=1)
    # print(score_df[score_df['score'] >= 600])
    return score_df


def scorecard_bins_woe(preprocess_result):
    numeric_vars = preprocess_result['numeric_vars']
    factor_vars = preprocess_result['factor_vars']

    numeric_vars_variable_and_lable_dict = {}
    factor_vars_variable_and_lable_dict = {}
    factor_vars_variable_and_lable = []
    vars_bin_woe_df = pd.DataFrame()
    # numeric数值型变量
    for numeric_var in numeric_vars:
        numeric_vars_bins_dict = dict()
        numeric_vars_woe_dict = dict()
        numeric_vars_variable_dict = dict()
        null_number = preprocess_result['numeric_breaks_dict'][numeric_var]['null_number']
        numeric_vars_bins_list = preprocess_result['numeric_breaks_dict'][numeric_var]['breaks_lst']
        include_miss_numeric_vars_bins_list = []
        # todo 对缺失值的标记
        if null_number != "None":
            for i in numeric_vars_bins_list:
                if i == null_number:
                    i = "null_flag"
                include_miss_numeric_vars_bins_list.append(i)
        if "null_flag" in include_miss_numeric_vars_bins_list:
            include_miss_numeric_vars_bins_list_ = include_miss_numeric_vars_bins_list[1:]
            numeric_vars_bins = [str(include_miss_numeric_vars_bins_list_[x:x + 2]) for x in
                                 range(len(include_miss_numeric_vars_bins_list_) - 1)]
            numeric_vars_bins.insert(0, "null_flag")
        else:
            numeric_vars_bins = [str(include_miss_numeric_vars_bins_list[x:x + 2]) for x in range(len(include_miss_numeric_vars_bins_list) - 1)]
        numeric_vars_breaks_labels = [k for k, v in preprocess_result['woe_dict'][numeric_var].items()]
        numeric_vars_woe = []
        numeric_vars_variable = []
        for label in numeric_vars_breaks_labels:
            numeric_vars_woe.append(preprocess_result['woe_dict'][numeric_var][label]['woe'])
            numeric_vars_variable.append(numeric_var + "_" + label)
            numeric_vars_variable_and_lable_dict.setdefault(numeric_var, []).append("_" + label)
        numeric_vars_bins_dict['bin'] = numeric_vars_bins
        numeric_vars_woe_dict['woe'] = numeric_vars_woe
        numeric_vars_variable_dict['variable'] = numeric_vars_variable
        numeric_var_dictdata = dict(dict(numeric_vars_variable_dict, **numeric_vars_bins_dict), **numeric_vars_woe_dict)
        numeric_var_df = pd.DataFrame(numeric_var_dictdata)
        # print(numeric_var_df)
        vars_bin_woe_df = vars_bin_woe_df.append(numeric_var_df)
    # factor字符类型变量
    for factor_var in factor_vars:
        factor_vars_bins_dict = dict()
        factor_vars_woe_dict = dict()
        factor_vars_variable_dict = dict()
        factor_var_bins_list = [str(k) for k, v in preprocess_result['woe_dict'][factor_var].items()]
        factor_var_woe_list = []
        factor_vars_variable = []
        for label in factor_var_bins_list:
            factor_var_woe_list.append(preprocess_result['woe_dict'][factor_var][label]['woe'])
            factor_vars_variable.append(factor_var + "_" + label)
            factor_vars_variable_and_lable_dict.setdefault(factor_var, []).append("_" + label)
            factor_vars_variable_and_lable.append("_" + label)
        factor_vars_bins_dict['bin'] = factor_var_bins_list
        factor_vars_woe_dict['woe'] = factor_var_woe_list
        factor_vars_variable_dict['variable'] = factor_vars_variable
        factor_var_dictdata = dict(dict(factor_vars_variable_dict, **factor_vars_bins_dict), **factor_vars_woe_dict)
        factor_var_df = pd.DataFrame(factor_var_dictdata)
        vars_bin_woe_df = vars_bin_woe_df.append(factor_var_df)
    vars_bin_woe_df = vars_bin_woe_df.reset_index(drop=True)
    return vars_bin_woe_df, factor_vars_variable_and_lable




























