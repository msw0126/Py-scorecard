# -*- coding:utf-8 -*-
import logging
import os
import colorlog
import datetime


def logger(log_file=str(datetime.date.today()) + ".log"):
    """
    打印和写入日志到文件
    Examples:
        logger().info("The training data were imported.")
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # 写入文件
    file_handler = logging.FileHandler(log_file, "a", "utf-8")
    file_handler.setLevel(logging.DEBUG)
    # 输出控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    # 定义handler的输出格式
    log_format = '\n'.join((
        '/' + '-' * 80,
        '[%(levelname)s][%(asctime)s][%(process)d:%(thread)d][%(filename)s:%(lineno)d %(funcName)s]:',
        '%(message)s',
        '-' * 80 + '/',
    ))
    color_log_format = '%(log_color)s' + log_format
    console_handler.setFormatter(colorlog.ColoredFormatter(color_log_format, log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red',
    }))
    file_handler.setFormatter(colorlog.ColoredFormatter(color_log_format, log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red',
    }))
    # 给logger添加handler
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def del_log(log_dir="./", n=7):
    """
    删除N天前的日志
    :param log_dir:
    :return:
    """
    # 获取两个日期间的所有日期
    def getEveryDay(begin_date,end_date):
        date_list = []
        begin_date = datetime.datetime.strptime(begin_date, "%Y-%m-%d")
        end_date = datetime.datetime.strptime(end_date,"%Y-%m-%d")
        while begin_date <= end_date:
            date_str = begin_date.strftime("%Y-%m-%d")
            date_list.append(date_str)
            begin_date += datetime.timedelta(days=1)
        return date_list
    for log_file in os.listdir(log_dir):
        if not log_file.endswith("log"):
            continue
        log_date = log_file.split(".")[0]
        today_date = datetime.date.today()
        if len(getEveryDay(str(log_date), str(today_date))) > n:
            os.remove(os.path.join(log_dir, log_file))

