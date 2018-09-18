"""
单个变量的分布图和分箱后的IV直方图。
调用的例子：
train_df, preprocess_conf_dict, preprocess_result = explore_preprocess_main(train_df, data_type_dict, explore_conf_dict)
folder = "D:\\TaoShu\\银行信贷业务信用风险管理\\xtx\\tmp_result"
plot_preprocess_result(preprocess_result, folder)
"""
from Utilities.Logger import logger as log
import matplotlib.pyplot as plt
import os


def plot_preprocess_result(preprocess_result, folder):
    """
    从输入中提取，转换出画图要用的数据，然后画图。
    从woe_dict里得到箱的正负样本比例，以及箱的大小，箱的woe，因子型变量的分箱规则在这里就可以得到
    从numeric_breaks_dict里得到数值型变量的分箱规则
    从iv_dict里得到每个变量的iv
    :param preprocess_result: csm的preprocess_result
    :param folder:
    :return: bin_names, count_list, proportion_list, IV, var_name
    """
    logger = log()
    logger.info("开始执行plot_preprocess_result()。")
    length = len(preprocess_result["woe_dict"])
    order = 1
    for var_name, tmp_dict in preprocess_result["woe_dict"].items():
        logger.info("总进度：%d/%d，正在处理变量：%s" % (order, length, var_name))
        order += 1
        IV = preprocess_result["iv_dict"][var_name]  # 该变量的iv值
        bin_names = list()
        count_list = list()
        proportion_list = list()
        if var_name in preprocess_result["numeric_breaks_dict"]:
            logger.info("%s是数值型变量" % var_name)
            num_dict = preprocess_result["numeric_breaks_dict"][var_name]
            break_list = num_dict["breaks_lst"]
            for ind, break_label in enumerate(num_dict["breaks_labels"]):
                name = "[%.2f, %.2f)" % (break_list[ind], break_list[ind + 1])
                count = tmp_dict[break_label]["nrow"]
                proportion = 1.0 * tmp_dict[break_label]["npos"] / tmp_dict[break_label]["nrow"]
                bin_names.append(name)
                count_list.append(count)
                proportion_list.append(proportion)
        else:
            logger.info("%s是因子型变量" % var_name )
            for k, v in tmp_dict.items():
                bin_names.append(k)
                count_list.append(v["nrow"])
                proportion_list.append(1.0 * v["npos"] / v["nrow"])  # 这个用箱内的y=1个数/箱的样本个数
        try:
            uni_analysis(bin_names, count_list, proportion_list, IV, var_name, folder)
        except Exception as e:
            logger.error("为变量%s画图时发生错误，跳过该变量的画图。\n"
                         "错误信息如下：\n"
                         "%s\n"
                         "画图参数是\n"
                         "bin_names: %s\n"
                         "count_list: %s\n"
                         "proportion_list: %s\n"
                         "IV: %f\n"
                         "var_name: %s\n"
                         "folder: %s"
                         "" % (var_name, e, str(bin_names), str(count_list), str(proportion_list), IV, var_name, folder))
            continue
    logger.info("plot_preprocess_result()执行完毕。")

def uni_analysis(bin_names, count_list, proportion_list, IV, var_name, folder):
    """
    画一个2*1的图。
    :param bin_names: 直方图的横坐标
    :param count_list: 直方图的纵坐标
    :param proportion_list: 直方图的纵坐标
    :param IV: 该变量的IV
    :param var_name: 该变量的名字
    :return: None
    """
    fig_width = 60
    fig_height = 40
    fontsize_xtick = 9 * 6
    fontsize_ytick = fontsize_xtick
    fontsize_ylabel = 9 * 6
    fontsize_title = 9 * 6
    rotation = 30
    text_xshift = -0.08

    def getshift(x):
        # 根据数据的长短决定位移的大小，比例位移固定用-0.08，分布的位移如下1对应-0.05，3对应-0.07
        len_x = len(str(x))
        return -0.01 * (len_x) - 0.04

    bar_width = 0.3 if len(bin_names) <= 10 else 0.2  # 柱子个数超过10，缩短柱子的宽度
    level_scale = {1: 1.0, 2: 0.85, 3: 0.7}
    level = 1
    for name in bin_names:
        # 如果有的箱的名字很长，则xtick使用更小的字体
        if len(name) > 14:
            level = 2
        if len(name) > 18:
            level = 3
    fontsize_xtick = int(level_scale[level] * fontsize_xtick)
    if not os.path.exists(folder):
        os.makedirs(folder)
    plot_name = os.path.join(folder, "".join(["UnivariateAnalysis_", var_name, ".png"]))
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(fig_width, fig_height))
    bar_x_locations = range(len(bin_names))
    plt.setp((ax1, ax2), xticks=bar_x_locations, xticklabels=bin_names)
    ax1.bar(bar_x_locations, count_list, bar_width)
    ax1.grid()
    ax1.set_title("Distribution", fontsize=fontsize_title)
    ax1.set_ylabel("Count", fontsize=fontsize_ylabel)
    ax1.xaxis.set_tick_params(labelsize=fontsize_xtick)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=rotation)
    ax1.yaxis.set_tick_params(labelsize=fontsize_ytick)
    for i, v in enumerate(count_list):
        ax1.text(i + getshift(v), v + 0.03 * max(count_list), str(v), color='blue', fontweight="bold",
                 fontsize=fontsize_ytick)
    analysis_name = "Performance Analysis(IV=%.4f)" % IV
    ax2.bar(bar_x_locations, proportion_list, bar_width)
    ax2.grid()
    ax2.set_title(analysis_name, fontsize=fontsize_title)
    ax2.set_ylabel("Prevalence(Proportion of Y=1 at each bin)", fontsize=int(0.8 * fontsize_ylabel))
    ax2.xaxis.set_tick_params(labelsize=fontsize_xtick)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=rotation)
    ax2.yaxis.set_tick_params(labelsize=fontsize_ytick)
    for i, v in enumerate(proportion_list):
        ax2.text(i + text_xshift, v + 0.03 * max(proportion_list), "%.2f" % (v), color='blue', fontweight="bold",
                 fontsize=fontsize_ytick)
    fig.suptitle(var_name, fontsize=fontsize_title)  # 图的总标题
    fig.tight_layout()  # 调整布局，避免出现遮挡
    fig.subplots_adjust(
        top=0.9)  # 必须一起使用, https://stackoverflow.com/questions/7066121/how-to-set-a-single-main-title-above-all-the-subplots-with-pyplot
    fig.savefig(plot_name)
