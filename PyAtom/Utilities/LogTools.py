from functools import wraps
import datetime


def time_decrator(process_name):
    def wrapper(func):
        @wraps(func)
        def _wrapper(*args, **kargs):
            start_time = datetime.datetime.now()
            print('********* %s start at: %s' % (process_name, str(start_time)))

            result = func(*args, **kargs)

            end_time = datetime.datetime.now()
            run_time = str(end_time - start_time)
            print('********* %s finished at: %s' % (process_name, end_time))
            print('********* %s total running time: %s \n' % (process_name, run_time))

            return result

        return _wrapper

    return wrapper


def time_recorder(last_time, stage, time_log_dict):
    """

    :param last_time: 
    :param stage: 
    :param time_log_dict: 
    :return: 
    """
    now_time = datetime.datetime.now()
    run_time = str((now_time - last_time))
    time_log_dict[stage] = str(run_time)

    return now_time, time_log_dict


def progress_display(step, total_steps, info_str, nbins=20):
    """

    :return: 
    """
    stage_tags = [int(1.0 * total_steps * (i + 1) / nbins) for i in range(nbins)]
    if step in stage_tags:
        print(info_str + ' finish: ' + str(round(100.0 * stage_tags.index(step) / nbins, 2)) + '%')

