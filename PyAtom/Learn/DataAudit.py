from Utilities.DataProcess import *
from Utilities.LogTools import *


@time_decrator('data  audit')
def explore_data_audit(data, explore_conf_dict, data_type_dict):
    """
    main function of auditing
    :param data:
    :param explore_conf_dict: configurations of exploring, dict.
    :return: a data object with imported and audited data as a dataframe in it
    """
    target_name = explore_conf_dict['target_varname']

    audit_result_dict = check_parameter(explore_conf_dict)
    audit_result_dict.update(check_type_dict(data_type_dict, target_name))
    audit_result_dict.update(check_vars(data, list(data_type_dict.keys())))
    # audit_result_dict.update(check_id(data_obj['df'], id_name, stage='explore'))
    audit_result_dict.update(check_target(data, target_name))
    audit_result_dict.update(check_data_vol(data, row_low_limit=100, col_low_limit=2))

    audit_result = False if False in list(audit_result_dict.values()) else True

    return audit_result, audit_result_dict

