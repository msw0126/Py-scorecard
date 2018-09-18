from Utilities.DataProcess import *
from Utilities.LogTools import *
from imblearn.over_sampling import SMOTE


@time_decrator('one hot encoding')
def one_hot_encoding(data_df, one_hot_vars):
    """
    
    :param data_df: 
    :param one_hot_vars: 
    :return: 
    """
    left_df = data_df[[i for i in data_df.columns if i not in one_hot_vars]]
    for var in one_hot_vars:
        data_df[var] = data_df[var].astype(str)

    data_one_hot_df = pd.get_dummies(data_df[one_hot_vars])
    var_names_after_encoding = data_one_hot_df.columns.tolist()
    data_df = pd.concat([left_df, data_one_hot_df], axis=1)

    return data_df, var_names_after_encoding


@time_decrator('unbalance cutoff')
def unbalance_cutoff(data_df, target_name, numeric_vars, unbalanced_cutoff):
    """
    under sampling
    :param data_df: target dataframe
    :param target_name: name of target variable (levels: 0, 1), str.
    :param numeric_vars:
    :param unbalanced_cutoff: cutoff value of unbalanced ratio (large / small, > 1), int or float
    :return: an undersampled dataframe
    """
    target_dict = data_df[target_name].value_counts().to_dict()
    label_0_am = int(target_dict[0])
    label_1_am = int(target_dict[1])
    sampled_method = None
    if label_0_am * 1.0 / label_1_am > unbalanced_cutoff:
        if label_1_am <= 100:
            cols = [i for i in data_df.columns if i != target_name]
            for col in numeric_vars:
                data_df[col] = data_df[col].astype(str)

            smote = SMOTE(ratio='minority')
            X, Y = smote.fit_sample(data_df[cols], data_df[target_name])
            data_df = pd.concat([X, Y], axis=1)

            for col in numeric_vars:
                data_df[col] = pd.to_numeric(data_df[col])

            sampled_method = 'smote'
        else:
            sample_num = int(label_1_am * unbalanced_cutoff)
            label_1_df = data_df[data_df[target_name] == 1]
            label_0_df = data_df[data_df[target_name] == 0]
            sampled_label_0_df = random_sampling(label_0_df, sub_num=sample_num)

            data_df = pd.concat([label_1_df, sampled_label_0_df]).sample(frac=1).reset_index(drop=True)

            sampled_method = 'undersampling'

    return data_df, sampled_method


@time_decrator('data transform')
def explore_data_transform(data_obj, explore_conf_dict):
    """
    main function of data transform
    :param data_obj: a data object with target dataframe in it
    :param explore_conf_dict: configurations of exploring, dict.
    :return: a data object with a transformed dataframe in it
    """
    target_name = explore_conf_dict['target_varname']
    numeric_vars = data_obj['info']['numeric_vars']
    factor_vars = data_obj['info']['factor_vars']

    # under sampling
    sampled_method = None
    if explore_conf_dict['unbalanced_cutoff']:
        unbalanced_cutoff = explore_conf_dict['unbalanced_cutoff']
        data_obj['df'], sampled_method = unbalance_cutoff(data_obj['df'][[target_name] + numeric_vars + factor_vars],
                                                          target_name, numeric_vars, unbalanced_cutoff)

    return data_obj, sampled_method
