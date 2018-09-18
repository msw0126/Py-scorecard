from Utilities.DataProcess import *


def act_data_audit(data_df, id_name, target_name, data_type_dict):
    """
    audit data
    :param data_df: a dataframe to audit
    :param id_name: id name of the data, str.
    :return: an audit result dict
    """
    audit_result_dict = dict()
    audit_result_dict.update(check_id(data_df, id_name))
    if target_name in list(data_df.columns):
        audit_result_dict.update(check_target(data_df, target_name))

    audit_result = False if False in list(audit_result_dict.values()) else True

    return audit_result, audit_result_dict
