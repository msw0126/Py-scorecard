
from Utilities.LogTools import *
from Learn.BinningAnalysis import binning_analysis_main
from Learn.CollinearityAnalysis import collinearity_analysis_main


@time_decrator('data analysis')
def explore_data_analysis(data_obj, explore_conf_dict):
    """
    main function of univariate analysis
    :param data_obj: a data object with target datafame in it
    :param explore_conf_dict: configurations of exploring, dict.
    :return: a data object with processed datafame during analysis in it
    """
    # binning and iv computing and iv cutoff
    data_obj = binning_analysis_main(data_obj, explore_conf_dict)
    # remove vars by collinearity
    data_obj = collinearity_analysis_main(data_obj, explore_conf_dict)

    return data_obj


