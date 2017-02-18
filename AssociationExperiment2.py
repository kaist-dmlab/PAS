import pickle
import pandas as pd
import numpy as np
import ruduce_category_features
import removed_default_value
import datetime as dt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from scipy import stats
# import get_classification_score as GetClassificationScore


def Entropy(values):
    value_cnt_dict = dict()
    for value in values:
        if value not in value_cnt_dict:
            value_cnt_dict[value] = 0
        value_cnt_dict[value] += 1
    value_ratio_list = list()
    for value in value_cnt_dict:
        value_ratio_list.append(value_cnt_dict[value] / float(sum(value_cnt_dict.values())))
    if len(value_ratio_list) == 1:
        entropy = 0.0
    else:
        entropy = stats.entropy(value_ratio_list) / np.log(len(value_ratio_list)) # Normalized
    return entropy, value_cnt_dict


def GetPatterns(mcpp_list, granularity_min):
    # Get periodic pattern for each length
    # Save unique pattern in each length
    length_mcpp_list = dict()
    for mcpp in mcpp_list:
        left_list = mcpp[0]
        right_list = mcpp[1]
        mcpp_len = len(left_list)
        mcpp_1_list = list()
        # print mcpp
        for mcpp_idx in range(mcpp_len):
            left_item = left_list[mcpp_idx]
            right_item = right_list[mcpp_idx]
            for item in left_item:
                if 'time_daily' in item:
                    time_str = item.split('time_daily:')[1]
                    time = dt.datetime.strptime(time_str, '%H:%M:%S')
                    break
            mcpp_1_list.append((time, left_item, right_item))
        if mcpp_len not in length_mcpp_list:
            length_mcpp_list[mcpp_len] = list()
        length_mcpp_list[mcpp_len].append(mcpp + (mcpp_1_list, ))

    # DONE: In the MCPP 1, the time must have only MCPP-1.
    new_mcpp_1_list = list()
    for mcpp1_info in length_mcpp_list[1]:
        mcpp1_time = mcpp1_info[-1][0][0]
        is_include = False
        for mcpp_len in length_mcpp_list:
            if mcpp_len != 1:
                for mcpp_info in length_mcpp_list[mcpp_len]:
                    mcpp_1_list = mcpp_info[-1]
                    mcpp_1_time_list = [mcpp_1[0] for mcpp_1 in mcpp_1_list]
                    if mcpp1_time in mcpp_1_time_list:
                        is_include = True
                        break
            if is_include:
                break
        if not is_include:
            new_mcpp_1_list.append(mcpp1_info)

    print "Get time ranges that contains only MCPP-1 (%d/%d)" % (len(new_mcpp_1_list), len(length_mcpp_list[1]))
    length_mcpp_list[1] = new_mcpp_1_list
    ##########################################################

    '''
    for length in length_mcpp_list:
        for mcpp in length_mcpp_list[length]:
            print "%s\t%s\t%s" % (mcpp[0], mcpp[1], str(mcpp[6]))
    '''
    # Find fully pattern and unfully pattern
    mcpp_len_list = sorted(length_mcpp_list.keys(), reverse=True)
    mcpp_1_list = length_mcpp_list[1]
    fully_unfully_pattern_list = list()
    for mcpp_len in mcpp_len_list:
        if mcpp_len == 1:
            continue
        for mcpp_info in length_mcpp_list[mcpp_len]:
            converted_mcpp_1_list = mcpp_info[-1]
            # Find mcpp that is same periodic pattern at the other time from 00:00:00 to 23:55:00
            time_diff = None
            start_mcpp_1 = converted_mcpp_1_list[0]
            start_mcpp_1_time = start_mcpp_1[0]
            start_time = dt.datetime.strptime('00:00:00', '%H:%M:%S')
            end_time = dt.datetime.strptime('00:00:00', '%H:%M:%S') + dt.timedelta(days=1) - dt.timedelta(minutes=granularity_min)
            # end_time = dt.datetime.strptime('23:30:00', '%H:%M:%S')
            time_diff = start_mcpp_1_time - start_time

            while True:
                has_unfully_pattern = False
                unfully_mcpp_list = list()
                for mcpp_1 in converted_mcpp_1_list:
                    mcpp1_time = mcpp_1[0]
                    converted_time = mcpp1_time - time_diff
                    if converted_time.year == 1899:
                        converted_time += dt.timedelta(days=1)

                    # print "-----------------"
                    # print mcpp1_time
                    # print time_diff
                    # print converted_time
                    # Change time in left pattern
                    mcpp_1 = list(mcpp_1)
                    left_item_list = list(mcpp_1[1])
                    for time_daily_idx, item in enumerate(left_item_list):
                        if 'time_daily' in item:
                            left_item_list[time_daily_idx] = "time_daily:%s" % (converted_time.strftime('%H:%M:%S'))
                            break

                    mcpp_1[0] = converted_time
                    mcpp_1[1] = tuple(left_item_list)
                    mcpp_1 = tuple(mcpp_1)

                    # Check converted pattern in fully pattenr. e.g., Fully: (11:00, 11:30), Unfully: (11:30, 12:00)
                    if mcpp_1[0] not in [converted_mcpp_1[0] for converted_mcpp_1 in converted_mcpp_1_list]:
                        is_include = False
                        converted_original_mcpp1_info = None

                        for sublength in sorted(length_mcpp_list.keys()):
                            if sublength == 1: # If wanna support unfully pattern has sub-sequence, remove this line.
                                sublength_mcpp_list = length_mcpp_list[sublength]
                                for origin_mcpp1_info in sublength_mcpp_list:
                                    origin_mcpp1 = origin_mcpp1_info[-1]
                                    # print mcpp_1
                                    # print origin_mcpp1
                                    if mcpp_1 in origin_mcpp1:
                                        is_include = True
                                        converted_original_mcpp1_info = origin_mcpp1_info
                                        break
                            if is_include:
                                break

                        if is_include:
                            has_unfully_pattern = True
                            unfully_mcpp_list.append(converted_original_mcpp1_info)
                        else:
                            has_unfully_pattern = False
                            break
                    else:
                        has_unfully_pattern = False
                        break

                if has_unfully_pattern:
                    # This is unfully pattern for the fully periodic pattern 'mcpp_info'
                    fully_unfully_pattern_list.append((mcpp_info, unfully_mcpp_list))
                time_diff -= dt.timedelta(minutes=granularity_min)
                if time_diff == 0:
                    time_diff -= dt.timedelta(minutes=granularity_min)
                last_converted_time = converted_mcpp_1_list[-1][0] - time_diff
                if last_converted_time > end_time:
                    break

    '''
    print "Conditional Fully Unfully"
    for fully_unfully_pattern in fully_unfully_pattern_list:
        print "%s\t%s" % (str(fully_unfully_pattern[0][:2]), str(fully_unfully_pattern[1]))
    '''

    return fully_unfully_pattern_list, length_mcpp_list

def IsContainInMCPP(single_mcpp, sequence_mcpp):
    sequence_len = len(sequence_mcpp[0])
    sequence_left = sequence_mcpp[0]
    sequence_right = sequence_mcpp[1]
    single_left = single_mcpp[0]
    single_right = single_mcpp[1]
    for idx in range(sequence_len):
        if single_left[0] == sequence_left[idx] and single_right[0] == sequence_right[idx]:
            return True
    return False


def PlotMCPPDistribution(length_mcpp_list, granularity_min, ax=None):
    #TODO: Get patterns for a each length
    #TODO: Count patterns for a each time range
    #TODO: Output, length-time-count

    length_dailytime_mcpp_dict = dict()
    for length in length_mcpp_list:
        for mcpp_info in length_mcpp_list[length]:
            mcpp_1_list = mcpp_info[-1]
            for mcpp_1 in mcpp_1_list:
                time_daily = mcpp_1[0]
                if length not in length_dailytime_mcpp_dict:
                    length_dailytime_mcpp_dict[length] = dict()
                if time_daily not in length_dailytime_mcpp_dict[length]:
                    length_dailytime_mcpp_dict[length][time_daily] = list()
                length_dailytime_mcpp_dict[length][time_daily].append(mcpp_info)

    '''
    for length in length_dailytime_mcpp_dict:
        for time_daily in length_dailytime_mcpp_dict[length]:
            for mcpp in length_dailytime_mcpp_dict[length][time_daily]:
                print "%d\t%s\t%s" % (length, time_daily, mcpp)
    '''

    start_timestamp = dt.datetime.strptime('00:00:00', '%H:%M:%S')
    time_range = [start_timestamp + x * dt.timedelta(minutes=granularity_min) for x in range(24 * 60 / granularity_min)]
    ind = np.arange(len(time_range))
    width = 0.35
    if ax is None:
        fig, ax = plt.subplots()
    label_info = (list(), list())
    length_list = sorted(length_dailytime_mcpp_dict.keys())
    time_mcpp_length_dict = dict()
    colors = iter(cm.rainbow(np.linspace(0, 1, len(length_list))))
    for length in length_list:
        y_list = list()
        for time in time_range:
            # time_str = time.strftime('%H:%M:%S')
            if time in length_dailytime_mcpp_dict[length]:
                mcpp_list = length_dailytime_mcpp_dict[length][time]
                y_list.append(len(mcpp_list))
            else:
                y_list.append(0)

        rects = ax.bar(ind + width * (length - 1), y_list, width, color=next(colors))
        label_info[0].append(rects[0])
        label_info[1].append("MCPP LEN %d" % length)

    ax.set_xticks(ind + width)
    ax.set_xticklabels([time.strftime('%H:%M') for time in time_range], rotation='vertical')
    ax.legend(label_info[0], label_info[1])
    ax.set_ylabel('# MCPP')

    if ax is None:
        plt.close()
    # plt.show()


def PlotClass(class_data, granularity_min, ax=None):

    dailytime_class_info_dict = dict()
    class_index_list = class_data.index
    for class_index in class_index_list:
        value = class_data[class_index]
        # print "%s\t%s" % (class_index, value)
        granularity_index = pd.DatetimeIndex(((np.round(pd.DatetimeIndex([class_index]).asi8 / (1e9 * 60 * granularity_min))) * 1e9 * 60 * granularity_min).astype(np.int64))[0]
        daily_time = granularity_index.strftime('%H:%M:%S')
        if daily_time not in dailytime_class_info_dict:
            dailytime_class_info_dict[daily_time] = list()
        dailytime_class_info_dict[daily_time].append((class_index, value))

    start_timestamp = dt.datetime.strptime('00:00:00', '%H:%M:%S')
    time_range = [start_timestamp + x * dt.timedelta(minutes=granularity_min) for x in range(24 * 60 / granularity_min)]

    if ax is None:
        fig, ax = plt.subplots()
    cnt_ax = ax
    y_avail_list = list()
    y_unavail_list = list()
    class_value_data = dict()
    daily_time_index_list = list()
    for value in list(set(class_data.values)):
        class_value_data[value] = list()
    for time in time_range:
        time_str = time.strftime('%H:%M:%S')
        available_cnt = 0
        unavailable_cnt = 0
        value_count_dict = dict()
        for value in list(set(class_data.values)):
            value_count_dict[value] = 0
        if time_str in dailytime_class_info_dict:
            class_list = dailytime_class_info_dict[time_str]
            for class_index, value in class_list:
                value_count_dict[value] += 1
                if value == 1:
                    available_cnt += 1
                elif value == 0:
                    unavailable_cnt += 1
        y_avail_list.append(available_cnt)
        y_unavail_list.append(unavailable_cnt)
        for value in class_value_data:
            class_value_data[value].append(value_count_dict[value])
        daily_time_index_list.append(time.strftime('%H:%M'))

    df = pd.DataFrame(data=class_value_data, index=daily_time_index_list, columns=class_value_data.keys())
    if len(class_value_data) > 0:
        df.plot(ax=cnt_ax, kind='bar', stacked=True)
    if ax is None:
        plt.close()


def PlotClassificationAccuracy(class_data, granularity_min, predict_actual_class_list, ax=None):
    daily_time_predict_true_false_dict = dict()
    class_index_list = class_data.index
    # print len(class_index_list)
    for idx, class_index in enumerate(class_index_list):
        value = class_data[class_index]
        granularity_index = pd.DatetimeIndex(((np.round(pd.DatetimeIndex([class_index]).asi8 / (1e9 * 60 * granularity_min))) * 1e9 * 60 * granularity_min).astype(np.int64))[0]
        daily_time = granularity_index.strftime('%H:%M:%S')
        print "%s\t%s\t%d\t%s\t%s" % (class_index, daily_time, value, predict_actual_class_list[idx][1], predict_actual_class_list[idx][0])
        if daily_time not in daily_time_predict_true_false_dict:
            daily_time_predict_true_false_dict[daily_time] = dict()
            daily_time_predict_true_false_dict[daily_time]['correct'] = list()
            daily_time_predict_true_false_dict[daily_time]['incorrect'] = list()

        if predict_actual_class_list[idx][1] == predict_actual_class_list[idx][0]:
            result_key = 'correct'
        else:
            result_key = 'incorrect'

        daily_time_predict_true_false_dict[daily_time][result_key].append((class_index, daily_time, value, predict_actual_class_list[idx][1], predict_actual_class_list[idx][0]))

    start_timestamp = dt.datetime.strptime('00:00:00', '%H:%M:%S')
    time_range = [start_timestamp + x * dt.timedelta(minutes=granularity_min) for x in range(24 * 60 / granularity_min)]
    ind = np.arange(len(time_range))
    width = 0.35
    # colors = ['r', 'g', 'b', 'c', 'y', 'm', 'k']
    if ax is None:
        fig, ax = plt.subplots()
    cnt_ax = ax
    y_list = list()
    for time in time_range:
        time_str = time.strftime('%H:%M:%S')
        accuracy = 0
        if time_str in daily_time_predict_true_false_dict:
            correct_list = daily_time_predict_true_false_dict[time_str]['correct']
            incorrect_list = daily_time_predict_true_false_dict[time_str]['incorrect']
            accuracy = len(correct_list) / float(len(correct_list) + len(incorrect_list))
        y_list.append(accuracy)

    avail_rects = cnt_ax.bar(ind, y_list, width, color='b')
    # unavail_rects = cnt_ax.bar(ind + width, y_unavail_list, width, color='y')

    cnt_ax.set_xticks(ind + width)
    cnt_ax.set_xticklabels([time.strftime('%H:%M') for time in time_range], rotation='vertical')
    # cnt_ax.legend((avail_rects[0], unavail_rects[0]), ("Available", "Unavailable"))
    cnt_ax.set_ylabel('Classification Accuracy')

    if ax is None:
        plt.close()

    return daily_time_predict_true_false_dict


def PlotEntropy(class_data, granularity_min, fully_unfully_pattern_list, ax=None, class_threshold=0):
    # (class_data, granularity_min, axarr[2], fully_unfully_pattern_list, class_threshold)
    dailytime_class_info_dict = dict()
    class_index_list = class_data.index
    for class_index in class_index_list:
        value = class_data[class_index]
        granularity_index = pd.DatetimeIndex(((np.round(pd.DatetimeIndex([class_index]).asi8 / (1e9 * 60 * granularity_min))) * 1e9 * 60 * granularity_min).astype(np.int64))[0]
        daily_time = granularity_index.strftime('%H:%M:%S')
        if daily_time not in dailytime_class_info_dict:
            dailytime_class_info_dict[daily_time] = list()
        dailytime_class_info_dict[daily_time].append((class_index, value))

    #TODO: Get fully pattern, unfully pattern.
    #TODO: Calculating entropy fully pattern and unfully pattern

    fully_unfully_entropy_list = list()
    for fully_unfully_pattern in fully_unfully_pattern_list:
        '''
        print fully_unfully_pattern
        print fully_unfully_pattern[0]
        print fully_unfully_pattern[1]
        '''
        fully_pattern = fully_unfully_pattern[0]
        mcpp_1_list = fully_pattern[-1]
        fully_class_list = list()
        for mcpp_1 in mcpp_1_list:
            tiemstamp = mcpp_1[0]
            daily_time = tiemstamp.strftime('%H:%M:%S')
            if daily_time in dailytime_class_info_dict:
                fully_class_list += dailytime_class_info_dict[daily_time]

        unfully_pattern_list = fully_unfully_pattern[1]
        unfully_class_list = list()
        for unfully_pattern in unfully_pattern_list:
            mcpp_1_list = unfully_pattern[-1]
            for mcpp_1 in mcpp_1_list:
                tiemstamp = mcpp_1[0]
                daily_time = tiemstamp.strftime('%H:%M:%S')
                if daily_time in dailytime_class_info_dict:
                    unfully_class_list += dailytime_class_info_dict[daily_time]

        class_value_list = list()
        for class_index, value in fully_class_list:
            class_value_list.append(value)
        fully_class_entropy = Entropy(class_value_list)[0]

        class_value_list = list()
        for class_index, value in unfully_class_list:
            class_value_list.append(value)
        unfully_class_entropy = Entropy(class_value_list)[0]

        '''
        print "----------------"
        print fully_pattern
        print unfully_pattern_list
        print fully_class_list
        print unfully_class_list
        print fully_class_entropy
        print unfully_class_entropy
        print "----------------"
        '''
        fully_unfully_entropy_list.append((fully_pattern, unfully_pattern_list, fully_class_list, unfully_class_list, fully_class_entropy, unfully_class_entropy))

    return fully_unfully_entropy_list


def ClassEntropyAnalysis(data, fully_unfully_pattern_list, length_mcpp_list, granularity_min, f_result, result_plot_path, class_threshold, class_label):
    # print class_label
    class_data = data[class_label].dropna()  # Because nan is in Sleeping time.

    fig, axarr = plt.subplots(3)
    PlotMCPPDistribution(length_mcpp_list, granularity_min, axarr[0])
    PlotClass(class_data, granularity_min, axarr[1])
    fully_unfully_entropy_list = PlotEntropy(class_data, granularity_min, fully_unfully_pattern_list, axarr[2], class_threshold)

    fig.set_size_inches(18, 18)
    plt.savefig(result_plot_path)
    # plt.show()
    plt.clf()
    plt.close(fig)

    return fully_unfully_entropy_list


def AnalysisEntropy(full_data, mcpp, file_path, class_threshold, granularity_min, class_label):
    result_file_path = 'entropy_result/%s_MCPP_%d_entropy_result_%s.txt' % (file_path.split('/')[-1].split('.pickle')[0][:10], granularity_min, class_label)
    result_plot_path = 'entropy_result/%s_MCPP_%d_entropy_%s.png' % (file_path.split('/')[-1].split('.pickle')[0][:10], granularity_min, class_label)
    f_result = open(result_file_path, 'w')
    fully_unfully_pattern_list, length_mcpp_list = GetPatterns(mcpp, granularity_min)
    fully_unfully_entropy_list = ClassEntropyAnalysis(full_data, fully_unfully_pattern_list, length_mcpp_list, granularity_min, f_result, result_plot_path, class_threshold, class_label)

    return fully_unfully_entropy_list


if __name__ == '__main__':
    dir_path = 'dataset/interpolated'
    output_dir = 'dataset/interpolated'
    granularity_min = 30
    file_info_list = ["c4836d99d240751347eb65fa265735f5bd96d291_timeseries_dict"]
    class_threshold = 20
    accuracy_threshold = 0.6
    '''
    for i, file_info in enumerate(file_info_list):
        print "%d/%d - %s" % (i, len(file_info_list), file_info)
        user, accuracy, auc, top_features, predict_actual_class_list = GetClassificationScore.GetClassificationScore(file_info, dir_path, output_dir, granularity_min=granularity_min)
        AnalysisEntropy(dir_path, output_dir, file_info, predict_actual_class_list, class_threshold, granularity_min, accuracy_threshold)
    '''
